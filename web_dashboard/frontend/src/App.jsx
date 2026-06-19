import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import TrackingView from './TrackingView'

const BACKEND_ADDRESS_KEY = 'fall-monitor-backend-address'
const SKELETON_VISIBILITY_KEY = 'fall-monitor-show-skeletons'
const isViteDevServer = ['5173', '5174', '5175'].includes(window.location.port)
const apiHost = window.location.hostname || '127.0.0.1'
const apiProtocol = window.location.protocol === 'https:' ? 'https' : 'http'
const localApiBaseUrl = isViteDevServer ? `${apiProtocol}://${apiHost}:8000` : window.location.origin
const normalizeLocalUrl = (value, fallback) => {
  if (!value) return fallback
  return value
    .replace('://127.0.0.1:', `://${apiHost}:`)
    .replace('://localhost:', `://${apiHost}:`)
}
const normalizeBackendAddress = (value, fallback = localApiBaseUrl) => {
  const raw = String(value || fallback || '').trim()
  const hasProtocol = /^[a-z]+:\/\//i.test(raw)
  const withProtocol = hasProtocol ? raw : `${apiProtocol}://${raw}`
  try {
    const url = new URL(withProtocol)
    if (['5173', '5174', '5175'].includes(url.port)) url.port = '8000'
    if (!url.port && /^(\d{1,3}\.){3}\d{1,3}$/.test(url.hostname)) url.port = '8000'
    url.pathname = ''
    url.search = ''
    url.hash = ''
    return url.origin
  } catch {
    return fallback
  }
}
const buildWsBaseUrl = (apiBaseUrl) => {
  const url = new URL(apiBaseUrl)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  url.pathname = '/ws'
  url.search = ''
  url.hash = ''
  return url.toString()
}
const queryBackendAddress = new URLSearchParams(window.location.search).get('server')
const storedBackendAddress = window.localStorage?.getItem(BACKEND_ADDRESS_KEY)
const API_BASE_URL = normalizeBackendAddress(
  queryBackendAddress || storedBackendAddress || normalizeLocalUrl(import.meta.env.VITE_API_BASE_URL, localApiBaseUrl),
)
if (queryBackendAddress) window.localStorage?.setItem(BACKEND_ADDRESS_KEY, API_BASE_URL)
const WS_BASE_URL = buildWsBaseUrl(API_BASE_URL)

const emptyRealtime = {
  points: [],
  tracks: [],
  skeletons: [],
  vitals: [],
  meta: {},
  timestamp: null,
}

const pageLabels = {
  overview: '관제 대시보드',
  monitor: '실시간 모니터',
  alerts: '알림 기록',
  settings: '시스템 설정',
}

const speedThresholdFromSlider = (value) => -1.5 + (Number(value) / 100) * 1.4
const heightThresholdFromSlider = (value) => 0.4 + (Number(value) / 100) * 1.1
const apiPath = (path) => `${API_BASE_URL}${path}`

const clamp = (value, min, max) => Math.min(max, Math.max(min, value))

const toNumber = (value, fallback = 0) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const formatTime = (value) => {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toLocaleString('ko-KR', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const formatPercent = (value) => {
  const score = toNumber(value, 0)
  return `${Math.round(score > 1 ? score : score * 100)}%`
}

const formatVector = (location) => {
  if (!Array.isArray(location) || location.length === 0) return '-'
  return location
    .slice(0, 3)
    .map((item) => toNumber(item).toFixed(2))
    .join(', ')
}

const visibleTrackId = (track, index, offset) => {
  const parsed = Number(track?.id)
  return Number.isFinite(parsed) ? Math.trunc(parsed) + offset : index + 1
}

const normalizeAlert = (payload) => {
  if (!payload) return null
  const timestamp = payload.timestamp || new Date().toISOString()
  return {
    id: payload.id ?? `${payload.track_id ?? payload.trackId ?? 'track'}-${timestamp}`,
    trackId: payload.track_id ?? payload.trackId ?? '-',
    score: toNumber(payload.score),
    location: Array.isArray(payload.location) ? payload.location : [],
    source: payload.source || 'radar',
    timestamp,
  }
}

const normalizeStatus = (payload) => ({
  backendConnected: Boolean(payload?.backendConnected ?? false),
  radarConnected: Boolean(payload?.radarConnected ?? payload?.running ?? false),
  activePeople: toNumber(payload?.activePeople ?? payload?.personCount ?? payload?.person_count, 0),
  frame: toNumber(payload?.frame, 0),
  pointCount: toNumber(payload?.pointCount, 0),
  firmwareObjects: toNumber(payload?.firmwareObjects ?? payload?.numDetectedObj, 0),
  vitalRecords: toNumber(payload?.vitalRecords, 0),
  visibleVitalRecords: toNumber(payload?.visibleVitalRecords, 0),
  fallbackVitalRecords: toNumber(payload?.fallbackVitalRecords, 0),
  invalidVitalRecords: toNumber(payload?.invalidVitalRecords, 0),
  malformedTlvs: toNumber(payload?.malformedTlvs, 0),
  lastHeartbeat: payload?.lastHeartbeat ?? payload?.timestamp ?? null,
  serverTime: payload?.serverTime ?? null,
  websocketClients: toNumber(payload?.websocketClients, 0),
})

const isFreshHeartbeat = (value, windowMs = 5000) => {
  if (!value) return false
  const time = new Date(value).getTime()
  return Number.isFinite(time) && Date.now() - time < windowMs
}

const mergeStatus = (current, payload) => {
  const next = normalizeStatus(payload)
  if (!next.radarConnected && current?.radarConnected && isFreshHeartbeat(current.lastHeartbeat)) {
    return {
      ...current,
      ...next,
      radarConnected: true,
      lastHeartbeat: current.lastHeartbeat,
    }
  }
  return next
}

function App() {
  const [status, setStatus] = useState(normalizeStatus(null))
  const [alerts, setAlerts] = useState([])
  const [blackboxFiles, setBlackboxFiles] = useState([])
  const [realtime, setRealtime] = useState(emptyRealtime)
  const [wsState, setWsState] = useState('disconnected')
  const [activePage, setActivePage] = useState('overview')
  const [speedSlider, setSpeedSlider] = useState(70)
  const [heightSlider, setHeightSlider] = useState(36)
  const [pendingAction, setPendingAction] = useState(null)
  const [settingsState, setSettingsState] = useState('대기')
  const [popupAlert, setPopupAlert] = useState(null)
  const [showSkeletons, setShowSkeletons] = useState(
    () => window.localStorage?.getItem(SKELETON_VISIBILITY_KEY) === '1',
  )
  const lastPopupKeyRef = useRef(null)
  const wsRef = useRef(null)

  const thresholds = useMemo(
    () => ({
      speed: speedThresholdFromSlider(speedSlider),
      height: heightThresholdFromSlider(heightSlider),
    }),
    [speedSlider, heightSlider],
  )

  const latestAlert = alerts[0] || null
  const isLive = status.radarConnected
  const connected = wsState === 'connected' || status.backendConnected
  const operationState = latestAlert ? 'attention' : isLive ? 'running' : 'idle'

  const showAlertPopup = useCallback((alert) => {
    if (!alert) return
    const key = `${alert.id}-${alert.timestamp}`
    if (lastPopupKeyRef.current === key) return
    lastPopupKeyRef.current = key
    setPopupAlert(alert)
  }, [])

  const mergeAlert = useCallback((payload) => {
    const alert = normalizeAlert(payload)
    if (!alert) return
    showAlertPopup(alert)
    setAlerts((current) => {
      const key = `${alert.id}-${alert.timestamp}`
      const withoutDuplicate = current.filter((item) => `${item.id}-${item.timestamp}` !== key)
      return [alert, ...withoutDuplicate].slice(0, 100)
    })
  }, [showAlertPopup])

  const handleSocketMessage = useCallback(
    (event) => {
      try {
        const payload = JSON.parse(event.data)
        const type = payload.type || 'unknown'

        if (type === 'STATUS') {
          setStatus((current) => mergeStatus(current, payload))
          return
        }

        if (type === 'ALERT') {
          mergeAlert(payload)
          return
        }

        if (type === 'realtime_data') {
          const meta = payload.meta && typeof payload.meta === 'object' ? payload.meta : {}
          setRealtime({
            points: Array.isArray(payload.points) ? payload.points : [],
            tracks: Array.isArray(payload.tracks) ? payload.tracks : [],
            skeletons: Array.isArray(payload.skeletons) ? payload.skeletons : [],
            vitals: Array.isArray(payload.vitals) ? payload.vitals : [],
            meta,
            timestamp: payload.timestamp || Date.now(),
          })
          setStatus((current) => normalizeStatus({
            ...current,
            ...meta,
            radarConnected: true,
            activePeople: payload.personCount ?? meta.personCount ?? payload.tracks?.length ?? current.activePeople,
            frame: meta.frameNumber ?? current.frame,
            pointCount: payload.pointCount ?? meta.pointCount ?? payload.points?.length ?? current.pointCount,
            lastHeartbeat: new Date().toISOString(),
          }))
        }
      } catch (error) {
        console.error('WebSocket message parse failed:', error)
      }
    },
    [mergeAlert],
  )

  const refreshSnapshot = useCallback(async () => {
    const [statusResult, alertsResult, blackboxResult] = await Promise.allSettled([
      fetch(apiPath('/api/status')),
      fetch(apiPath('/api/alerts')),
      fetch(apiPath('/api/blackbox/list')),
    ])

    if (statusResult.status === 'fulfilled' && statusResult.value.ok) {
      const payload = await statusResult.value.json()
      setStatus((current) => mergeStatus(current, payload))
    }

    if (alertsResult.status === 'fulfilled' && alertsResult.value.ok) {
      const rows = await alertsResult.value.json()
      if (Array.isArray(rows)) {
        const nextAlerts = rows.map(normalizeAlert).filter(Boolean)
        setAlerts(nextAlerts)
        if (nextAlerts.length > 0) showAlertPopup(nextAlerts[0])
      }
    }

    if (blackboxResult.status === 'fulfilled' && blackboxResult.value.ok) {
      const rows = await blackboxResult.value.json()
      if (Array.isArray(rows)) setBlackboxFiles(rows.slice(0, 12))
    }
  }, [showAlertPopup])

  useEffect(() => {
    let closedByEffect = false
    let reconnectTimer = null

    const connect = () => {
      setWsState('connecting')
      const socket = new WebSocket(WS_BASE_URL)
      wsRef.current = socket

      socket.onopen = () => {
        setWsState('connected')
        socket.send(JSON.stringify({ type: 'DASHBOARD_INIT' }))
      }
      socket.onmessage = handleSocketMessage
      socket.onerror = () => setWsState('error')
      socket.onclose = () => {
        setWsState('disconnected')
        if (!closedByEffect) reconnectTimer = window.setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      closedByEffect = true
      window.clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [handleSocketMessage])

  useEffect(() => {
    refreshSnapshot()
    const timer = window.setInterval(refreshSnapshot, 2500)
    return () => window.clearInterval(timer)
  }, [refreshSnapshot])

  useEffect(() => {
    window.localStorage?.setItem(SKELETON_VISIBILITY_KEY, showSkeletons ? '1' : '0')
  }, [showSkeletons])

  useEffect(() => {
    if (!popupAlert) return undefined
    const timer = window.setTimeout(() => setPopupAlert(null), 6000)
    return () => window.clearTimeout(timer)
  }, [popupAlert])

  const sendControl = async (command) => {
    setPendingAction(command)
    try {
      const response = await fetch(apiPath('/api/control'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setStatus((current) => ({ ...current, radarConnected: command === 'start' }))
    } catch (error) {
      console.error('Control request failed:', error)
    } finally {
      setPendingAction(null)
    }
  }

  const saveSettings = async () => {
    setSettingsState('전송 중')
    try {
      const response = await fetch(apiPath('/api/settings'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: thresholds.speed, height: thresholds.height }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setSettingsState('저장됨')
    } catch (error) {
      console.error('Settings request failed:', error)
      setSettingsState('전송 실패')
    }
  }

  const exportAlerts = () => {
    const blob = new Blob([JSON.stringify({ exportedAt: new Date().toISOString(), alerts }, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `fall-alerts-${Date.now()}.json`
    document.body.appendChild(anchor)
    anchor.click()
    document.body.removeChild(anchor)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="app-shell">
      <main className="main">
        {activePage !== 'overview' && <HospitalHeader activePage={activePage} setActivePage={setActivePage} />}

        {activePage === 'overview' && (
          <OverviewPage
            alerts={alerts}
            connected={connected}
            isLive={isLive}
            latestAlert={latestAlert}
            operationState={operationState}
            pendingAction={pendingAction}
            realtime={realtime}
            sendControl={sendControl}
            setActivePage={setActivePage}
            status={status}
          />
        )}

        {activePage === 'monitor' && (
          <MonitorPage
            connected={connected}
            latestAlert={latestAlert}
            realtime={realtime}
            setShowSkeletons={setShowSkeletons}
            showSkeletons={showSkeletons}
            status={status}
          />
        )}

        {activePage === 'alerts' && (
          <AlertsPage alerts={alerts} blackboxFiles={blackboxFiles} exportAlerts={exportAlerts} />
        )}

        {activePage === 'settings' && (
          <SettingsPage
            apiBaseUrl={API_BASE_URL}
            heightSlider={heightSlider}
            saveSettings={saveSettings}
            setHeightSlider={setHeightSlider}
            setSettingsState={setSettingsState}
            setShowSkeletons={setShowSkeletons}
            setSpeedSlider={setSpeedSlider}
            settingsState={settingsState}
            showSkeletons={showSkeletons}
            speedSlider={speedSlider}
            thresholds={thresholds}
            wsBaseUrl={WS_BASE_URL}
          />
        )}
      </main>
      {popupAlert && (
        <div className="fall-popup" role="alert">
          <strong>낙상 알림</strong>
          <span>트랙 {popupAlert.trackId} · 점수 {formatPercent(popupAlert.score)}</span>
          <small>{formatTime(popupAlert.timestamp)}</small>
          <button aria-label="알림 닫기" onClick={() => setPopupAlert(null)} type="button">
            x
          </button>
        </div>
      )}
    </div>
  )
}

function HospitalHeader({ activePage, setActivePage }) {
  return (
    <header className="hospital-header">
      <HospitalLogoButton setActivePage={setActivePage} />
      <div className="screen-title">
        <h1>{pageLabels[activePage] || pageLabels.overview}</h1>
      </div>
    </header>
  )
}

function HospitalLogoButton({ setActivePage }) {
  return (
    <button
      className="hospital-logo"
      onClick={() => setActivePage('overview')}
      title="관제 대시보드"
      type="button"
    >
      <svg className="logo-mark" aria-hidden="true" viewBox="0 0 44 44">
        <rect className="logo-base" x="2" y="2" width="40" height="40" rx="12" />
        <path className="logo-cross" d="M22 12v20M12 22h20" />
        <path className="logo-pulse" d="M12 30h6l3-6 4 10 3-4h4" />
      </svg>
      <span className="logo-copy">
        <strong>00 병원 낙상 감지 시스템</strong>
      </span>
    </button>
  )
}

function OverviewPage({
  alerts,
  connected,
  isLive,
  latestAlert,
  operationState,
  pendingAction,
  realtime,
  sendControl,
  setActivePage,
  status,
}) {
  const title =
    operationState === 'attention' ? '확인이 필요한 알림이 있습니다' : isLive ? '레이더 감지 중입니다' : '감시 대기 상태입니다'

  return (
    <div className="overview-page">
      <section className={`status-panel ${operationState}`}>
        <div className="status-copy">
          <div className="status-meta-row">
            <HospitalLogoButton setActivePage={setActivePage} />
            <span className="state-badge">{operationState === 'attention' ? '주의' : isLive ? '감시 중' : '대기'}</span>
          </div>
          <h2>{title}</h2>
          <p>
            {latestAlert
              ? `트랙 ${latestAlert.trackId} · 점수 ${formatPercent(latestAlert.score)} · ${formatTime(latestAlert.timestamp)}`
              : `서버 ${connected ? '연결됨' : '대기'} · 활성 인원 ${status.activePeople}명 · 프레임 ${status.frame}`}
          </p>
        </div>
        <div className="status-actions">
          <button
            className={isLive ? 'button danger' : 'button primary'}
            disabled={Boolean(pendingAction)}
            onClick={() => sendControl(isLive ? 'stop' : 'start')}
            type="button"
          >
            {pendingAction ? '처리 중' : isLive ? '감시 중지' : '감시 시작'}
          </button>
          <button
            aria-label="시스템 설정"
            className="icon-button"
            onClick={() => setActivePage('settings')}
            title="시스템 설정"
            type="button"
          >
            ⚙
          </button>
        </div>
      </section>

      <section className="metric-grid">
        <MetricCard label="활성 인원" value={`${status.activePeople}명`} />
        <MetricCard label="추적 대상" value={realtime.tracks.length} />
        <MetricCard label="포인트" value={realtime.points.length} />
        <MetricCard label="바이탈" value={realtime.vitals.length || status.visibleVitalRecords || status.vitalRecords} />
        <MetricCard label="알림" value={alerts.length} tone="danger" />
      </section>

      <section
        aria-label="실시간 모니터 열기"
        className="panel map-panel interactive-panel"
        onClick={() => setActivePage('monitor')}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            setActivePage('monitor')
          }
        }}
        role="button"
        tabIndex={0}
      >
        <PanelHeader title="실시간 위치" value={isLive ? '감지 중' : '대기'} />
        <MapPreview latestAlert={latestAlert} realtime={realtime} />
      </section>

      <section className="panel recent-panel">
        <PanelHeader
          action={<button onClick={() => setActivePage('alerts')} type="button">전체 보기</button>}
          title="최근 알림"
        />
        <AlertList alerts={alerts.slice(0, 4)} />
      </section>
    </div>
  )
}

function MonitorPage({ connected, realtime, status, latestAlert, showSkeletons, setShowSkeletons }) {
  return (
    <div className="monitor-page">
      <TrackingView realtime={realtime} connected={connected} showSkeletons={showSkeletons} />
      <aside className="monitor-side">
        <section className="panel">
          <PanelHeader title="Display" value={showSkeletons ? 'Skeleton ON' : 'Skeleton OFF'} />
          <button className="button secondary full-width" onClick={() => setShowSkeletons((value) => !value)} type="button">
            {showSkeletons ? 'Hide skeleton' : 'Show skeleton'}
          </button>
        </section>

        <section className="panel">
          <PanelHeader title="라이브 상태" value={connected ? '연결됨' : '오프라인'} />
          <InfoRow label="Backend" value={status.backendConnected ? 'OK' : 'No response'} />
          <InfoRow label="Web clients" value={status.websocketClients} />
          <InfoRow label="활성 인원" value={`${status.activePeople}명`} />
          <InfoRow label="프레임" value={status.frame} />
          <InfoRow label="포인트" value={realtime.points.length} />
          <InfoRow label="펌웨어 객체" value={status.firmwareObjects} />
          <InfoRow label="바이탈" value={`${realtime.vitals.length || status.visibleVitalRecords}/${status.vitalRecords}`} />
          <InfoRow label="마지막 수신" value={formatTime(status.lastHeartbeat)} />
        </section>

        <section className="panel">
          <PanelHeader title="현재 트랙" value={realtime.tracks.length} />
          <TrackList tracks={realtime.tracks} />
        </section>

        <section className="panel">
          <PanelHeader title="심박/호흡" value={realtime.vitals.length || status.visibleVitalRecords} />
          <VitalList vitals={realtime.vitals} />
        </section>

        {latestAlert && (
          <section className="alert-note">
            <span>최근 알림</span>
            <strong>트랙 {latestAlert.trackId}</strong>
            <small>{formatPercent(latestAlert.score)} · {formatTime(latestAlert.timestamp)}</small>
          </section>
        )}
      </aside>
    </div>
  )
}

function AlertsPage({ alerts, blackboxFiles, exportAlerts }) {
  return (
    <div className="records-page">
      <section className="panel records-main">
        <PanelHeader
          action={<button disabled={alerts.length === 0} onClick={exportAlerts} type="button">내보내기</button>}
          title="낙상 알림 기록"
          value={`${alerts.length}건`}
        />
        <AlertList alerts={alerts} large />
      </section>

      <section className="panel">
        <PanelHeader title="블랙박스" value={`${blackboxFiles.length}개`} />
        <div className="file-list">
          {blackboxFiles.length === 0 ? (
            <EmptyState text="저장된 파일이 없습니다." />
          ) : (
            blackboxFiles.map((fileName) => (
              <a href={apiPath(`/api/blackbox/${encodeURIComponent(fileName)}`)} key={fileName}>
                <span>{fileName}</span>
                <strong>열기</strong>
              </a>
            ))
          )}
        </div>
      </section>
    </div>
  )
}

function SettingsPage({
  apiBaseUrl,
  heightSlider,
  saveSettings,
  setHeightSlider,
  setSettingsState,
  setShowSkeletons,
  setSpeedSlider,
  settingsState,
  showSkeletons,
  speedSlider,
  thresholds,
  wsBaseUrl,
}) {
  return (
    <div className="settings-page">
      <section className="panel settings-card">
        <PanelHeader title="감지 민감도" value={settingsState} />
        <SliderControl
          label="속도 임계값"
          max="100"
          min="0"
          onChange={(value) => {
            setSpeedSlider(value)
            setSettingsState('변경됨')
          }}
          suffix="m/s"
          value={speedSlider}
          visibleValue={thresholds.speed.toFixed(2)}
        />
        <SliderControl
          label="높이 임계값"
          max="100"
          min="0"
          onChange={(value) => {
            setHeightSlider(value)
            setSettingsState('변경됨')
          }}
          suffix="m"
          value={heightSlider}
          visibleValue={thresholds.height.toFixed(2)}
        />
        <button className="button primary settings-save" onClick={saveSettings} type="button">
          설정 전송
        </button>
      </section>

      <section className="panel">
        <PanelHeader title="연결 정보" />
        <InfoRow label="API" value={apiBaseUrl} />
        <InfoRow label="WebSocket" value={wsBaseUrl} />
      </section>
    </div>
  )
}

function MetricCard({ label, value, tone = 'normal' }) {
  return (
    <article className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  )
}

function PanelHeader({ action, title, value }) {
  return (
    <div className="panel-header">
      <h3>{title}</h3>
      {action || (value ? <span>{value}</span> : null)}
    </div>
  )
}

function MapPreview({ latestAlert, realtime }) {
  const tracks = Array.isArray(realtime?.tracks) ? realtime.tracks.slice(0, 8) : []
  const latestTrackId = latestAlert ? String(latestAlert.trackId) : null
  const idOffset = tracks.some((track) => Number(track?.id) === 0) ? 1 : 0

  return (
    <div className="map-frame">
      <div className="map-meta">
          <span>레이더 01</span>
          <span>포인트 {Array.isArray(realtime?.points) ? realtime.points.length : 0}</span>
      </div>
      <div className="floor-map">
        <span className="radar-pin" />
        <span className="range-line vertical" />
        <span className="range-line horizontal" />
        {tracks.map((track, index) => {
          const position = Array.isArray(track.pos) ? track.pos : track.location
          const x = clamp(50 + toNumber(position?.[0], (index % 4) - 1.5) * 10, 8, 92)
          const y = clamp(50 - toNumber(position?.[1], Math.floor(index / 4) - 0.5) * 10, 8, 92)
          const fall = String(track.state || '').toUpperCase().includes('FALL') || String(track.id) === latestTrackId
          const displayId = visibleTrackId(track, index, idOffset)

          return (
            <span
              className={fall ? 'map-person fall' : 'map-person'}
              key={track.id ?? index}
              style={{ left: `${x}%`, top: `${y}%` }}
            >
              {displayId}
            </span>
          )
        })}
        {tracks.length === 0 && <div className="map-empty">감지 데이터 대기</div>}
      </div>
    </div>
  )
}

function AlertList({ alerts, large = false }) {
  if (alerts.length === 0) return <EmptyState text="표시할 알림이 없습니다." />

  return (
    <div className={large ? 'alert-list large' : 'alert-list'}>
      {alerts.map((alert) => (
        <article className="alert-row" key={`${alert.id}-${alert.timestamp}`}>
          <div>
            <strong>트랙 {alert.trackId}</strong>
            <span>{formatTime(alert.timestamp)}</span>
          </div>
          <p>점수 {formatPercent(alert.score)} · 위치 {formatVector(alert.location)} · {alert.source}</p>
        </article>
      ))}
    </div>
  )
}

function TrackList({ tracks }) {
  if (!tracks.length) return <EmptyState text="추적 중인 대상이 없습니다." />

  return (
    <div className="track-list">
      {tracks.map((track, index) => {
        const isFall = String(track.state || '').toUpperCase().includes('FALL')
        const idOffset = tracks.some((item) => Number(item?.id) === 0) ? 1 : 0
        const displayId = visibleTrackId(track, index, idOffset)
        return (
          <article className={isFall ? 'track-row fall' : 'track-row'} key={track.id ?? index}>
            <div>
              <strong>ID {displayId}</strong>
              <span>{formatTrackState(track.state)}</span>
            </div>
            <p>위치 {formatVector(track.pos)}</p>
            <p>포인트 {toNumber(track.pointCount, 0)} · 신뢰도 {formatPercent(track.confidence || 0)}</p>
            {track.vital && (
              <p className="vital-inline">
                HR {formatBpm(track.vital.heartBpm)} · RR {formatBpm(track.vital.breathBpm)}
              </p>
            )}
          </article>
        )
      })}
    </div>
  )
}

function VitalList({ vitals }) {
  if (!Array.isArray(vitals) || vitals.length === 0) {
    return <EmptyState text="표시할 바이탈 기록이 없습니다." />
  }

  return (
    <div className="vital-list">
      {vitals.map((vital, index) => (
        <article className={vital.fallback ? 'vital-row fallback' : 'vital-row'} key={`${vital.id ?? index}-${vital.rangeBin ?? 0}`}>
          <div>
            <strong>ID {vital.id ?? index + 1}</strong>
            <span>{vital.fallback ? 'fallback' : `range ${vital.rangeBin ?? '-'}`}</span>
          </div>
          <p>HR {formatBpm(vital.heartBpm)} · RR {formatBpm(vital.breathBpm)}</p>
          <small>deviation {toNumber(vital.deviation, 0).toFixed(3)}</small>
        </article>
      ))}
    </div>
  )
}

function InfoRow({ label, value }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function formatTrackState(state) {
  const label = String(state || '').toUpperCase()
  if (label.includes('FALL')) return '낙상'
  if (label.includes('GROUND') || label.includes('LOW')) return '낮은 자세'
  if (label.includes('LOST')) return '추적 손실'
  return '정상'
}

function formatBpm(value) {
  const bpm = toNumber(value, 0)
  return bpm > 0 ? `${bpm.toFixed(1)} bpm` : '-'
}

function SliderControl({ label, max, min, onChange, suffix, value, visibleValue }) {
  return (
    <label className="slider-control">
      <span>{label}</span>
      <strong>{visibleValue} {suffix}</strong>
      <input
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
        type="range"
        value={value}
      />
    </label>
  )
}

function EmptyState({ text }) {
  return <div className="empty-state">{text}</div>
}

export default App

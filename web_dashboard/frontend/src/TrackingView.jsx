import { Canvas } from '@react-three/fiber'
import { Grid, OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'
import { useMemo } from 'react'

const safeVector = (value, fallback) => {
  if (!Array.isArray(value)) return fallback
  return fallback.map((item, index) => {
    const parsed = Number(value[index])
    return Number.isFinite(parsed) ? parsed : item
  })
}

const stateColor = (state) => {
  const label = String(state || '').toUpperCase()
  if (label.includes('FALL')) return '#ef4444'
  if (label.includes('GROUND') || label.includes('LOW')) return '#f59e0b'
  if (label.includes('LOST')) return '#6b7280'
  return '#38bdf8'
}

const stateLabel = (state) => {
  const label = String(state || '').toUpperCase()
  if (label.includes('FALL')) return '낙상'
  if (label.includes('GROUND') || label.includes('LOW')) return '낮은 자세'
  if (label.includes('LOST')) return '추적 손실'
  return '정상'
}

const visibleTrackId = (track, index, offset) => {
  const parsed = Number(track?.id)
  return Number.isFinite(parsed) ? Math.trunc(parsed) + offset : index + 1
}

function PointCloud({ points }) {
  const positions = useMemo(() => {
    if (!Array.isArray(points) || points.length === 0) return new Float32Array(0)
    const values = []
    points.forEach((point) => {
      if (!Array.isArray(point) || point.length < 3) return
      const x = Number(point[0])
      const y = Number(point[1])
      const z = Number(point[2])
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        values.push(x, y, z)
      }
    })
    return new Float32Array(values)
  }, [points])

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color="#38bdf8" size={0.075} transparent opacity={0.72} sizeAttenuation />
    </points>
  )
}

function SkeletonLines({ skeletons }) {
  const positions = useMemo(() => {
    if (!Array.isArray(skeletons) || skeletons.length === 0) return new Float32Array(0)
    const values = []
    for (let offset = 0; offset < skeletons.length; offset += 12) {
      const chunk = skeletons.slice(offset, offset + 12)
      const parsed = chunk
        .map((point) => {
          if (!Array.isArray(point) || point.length < 3) return null
          const x = Number(point[0])
          const y = Number(point[1])
          const z = Number(point[2])
          return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) ? [x, y, z] : null
        })
        .filter(Boolean)

      if (parsed.length !== 12) continue

      const zValues = parsed.map((point) => point[2])
      const zSpan = Math.max(...zValues) - Math.min(...zValues)
      const maxSegment = parsed.reduce((longest, point, index) => {
        if (index % 2 === 1) return longest
        const next = parsed[index + 1]
        if (!next) return longest
        const length = Math.hypot(point[0] - next[0], point[1] - next[1], point[2] - next[2])
        return Math.max(longest, length)
      }, 0)

      if (zSpan < 0.8 || maxSegment > 2.4) continue

      for (const point of parsed) {
        values.push(point[0], point[1], point[2])
      }
    }
    return new Float32Array(values)
  }, [skeletons])

  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#60a5fa" transparent opacity={0.82} />
    </lineSegments>
  )
}

function TrackBox({ track, index, idOffset }) {
  const position = safeVector(track.pos, [0, 0, 0.8])
  const dims = safeVector(track.dims, [0.5, 0.5, 1.6]).map((value) => Math.max(0.08, value))
  const color = stateColor(track.state)
  const displayId = visibleTrackId(track, index, idOffset)

  return (
    <group position={position}>
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(dims[0], dims[1], dims[2])]} />
        <lineBasicMaterial color={color} />
      </lineSegments>
      <Text position={[0, 0, dims[2] / 2 + 0.18]} fontSize={0.18} color={color} anchorX="center" anchorY="middle">
        {`ID ${displayId}\n${stateLabel(track.state)}`}
      </Text>
    </group>
  )
}

function TrackingView({ realtime, connected, showSkeletons = false }) {
  const points = Array.isArray(realtime?.points) ? realtime.points : []
  const tracks = Array.isArray(realtime?.tracks) ? realtime.tracks : []
  const skeletons = Array.isArray(realtime?.skeletons) ? realtime.skeletons : []
  const idOffset = tracks.some((track) => Number(track?.id) === 0) ? 1 : 0

  return (
    <section className="tracking-shell">
      <div className="viewer-status">
        <span className={connected ? 'live' : 'offline'}>{connected ? '연결됨' : '오프라인'}</span>
        <span>포인트 {points.length}</span>
        <span>트랙 {tracks.length}</span>
      </div>
      <Canvas camera={{ position: [3.3, -5.2, 3.5], fov: 48, up: [0, 0, 1] }}>
        <color attach="background" args={['#101214']} />
        <ambientLight intensity={0.65} />
        <directionalLight position={[4, -4, 6]} intensity={1.4} />
        <OrbitControls
          makeDefault
          target={[0, 0, 0.9]}
          enableDamping
          dampingFactor={0.08}
          minDistance={1.5}
          maxDistance={18}
          maxPolarAngle={Math.PI / 2 - 0.04}
        />
        <Grid
          position={[0, 0, 0]}
          rotation={[Math.PI / 2, 0, 0]}
          args={[14, 14]}
          cellSize={0.5}
          cellThickness={0.7}
          cellColor="#384047"
          sectionSize={2}
          sectionThickness={1.2}
          sectionColor="#5b6470"
          fadeDistance={18}
          fadeStrength={1.4}
          infiniteGrid
        />
        <axesHelper args={[1.2]} />
        <PointCloud points={points} />
        {showSkeletons && <SkeletonLines skeletons={skeletons} />}
        {tracks.map((track, index) => (
          <TrackBox idOffset={idOffset} track={track} index={index} key={track.id ?? index} />
        ))}
      </Canvas>
    </section>
  )
}

export default TrackingView

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const http = require('http');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 8000;
const ROOT_DIR = path.resolve(__dirname, '..', '..');
const DEFAULT_BLACKBOX_DIR = path.join(ROOT_DIR, '낙상감지기_데스크탑앱', 'records');
const BLACKBOX_DIR = process.env.FALL_BLACKBOX_DIR || DEFAULT_BLACKBOX_DIR;
const FRONTEND_DIST = path.resolve(__dirname, '..', 'frontend', 'dist');

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Create HTTP server
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    console.log('[WebSocket] Client connected');

    ws.on('message', (message) => {
        const messageString = message.toString();
        try {
            const data = JSON.parse(messageString);

            // Handle STATUS message from Python
            if (data.type === 'STATUS') {
                console.log('[WebSocket] Received STATUS:', data);
                const incomingConnected = data.running ?? data.radarConnected;
                if (incomingConnected === true) {
                    lastControlCommand = 'start';
                }
                const realtimeFresh = lastRealtimeAt && (Date.now() - lastRealtimeAt < 5000);
                const keepLiveFromRealtime = incomingConnected === false && realtimeFresh && lastControlCommand !== 'stop';
                systemStatus = {
                    ...systemStatus,
                    ...data,
                    radarConnected: keepLiveFromRealtime ? true : Boolean(incomingConnected),
                    activePeople: data.personCount ?? data.activePeople ?? systemStatus.activePeople,
                    lastHeartbeat: new Date().toISOString()
                };
            }

            // Handle REALTIME_DATA message from Python (3D Visualization)
            if (data.type === 'realtime_data') {
                const pointCount = data.points ? data.points.length : 0;
                console.log(`[WebSocket] Received DATA: ${pointCount} points`);
                latestRealtime = {
                    ...data,
                    receivedAt: new Date().toISOString()
                };
                lastRealtimeAt = Date.now();
                lastControlCommand = 'start';
                systemStatus = {
                    ...systemStatus,
                    radarConnected: true,
                    activePeople: data.personCount ?? data.meta?.personCount ?? data.tracks?.length ?? 0,
                    frame: data.meta?.frameNumber ?? systemStatus.frame ?? 0,
                    pointCount: data.pointCount ?? data.meta?.pointCount ?? pointCount,
                    firmwareObjects: data.meta?.numDetectedObj ?? data.meta?.firmwareObjects ?? systemStatus.firmwareObjects ?? 0,
                    vitalRecords: data.vitalCount ?? data.meta?.vitalRecords ?? systemStatus.vitalRecords ?? 0,
                    visibleVitalRecords: data.meta?.visibleVitalRecords ?? systemStatus.visibleVitalRecords ?? 0,
                    fallbackVitalRecords: data.meta?.fallbackVitalRecords ?? systemStatus.fallbackVitalRecords ?? 0,
                    invalidVitalRecords: data.meta?.invalidVitalRecords ?? systemStatus.invalidVitalRecords ?? 0,
                    malformedTlvs: data.meta?.malformedTlvs ?? systemStatus.malformedTlvs ?? 0,
                    lastHeartbeat: new Date().toISOString()
                };
            }

            // Handle ALERT message from Python
            if (data.type === 'ALERT') {
                console.log('[WebSocket] Received ALERT:', data);
                const newAlert = {
                    ...data,
                    timestamp: new Date().toISOString()
                };
                latestAlert = newAlert;
                alertHistory.unshift(newAlert);
                if (alertHistory.length > 1000) alertHistory.pop();
            }
        } catch (e) {
            // Not JSON or other error, ignore
        }

        // Broadcast received message to all OTHER connected clients
        wss.clients.forEach((client) => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
                client.send(messageString);
            }
        });
    });

    ws.on('close', () => {
        console.log('[WebSocket] Client disconnected');
    });
});

// Store latest alert and status
let latestAlert = null;
let latestRealtime = null;
let lastRealtimeAt = null;
let lastControlCommand = null;
let alertHistory = [];
let systemStatus = {
    radarConnected: false,
    activePeople: 0,
    frame: 0,
    pointCount: 0,
    firmwareObjects: 0,
    vitalRecords: 0,
    visibleVitalRecords: 0,
    fallbackVitalRecords: 0,
    invalidVitalRecords: 0,
    malformedTlvs: 0,
    lastHeartbeat: null
};
let systemCommand = null; // 'start' or 'stop'

// API Endpoint to receive alerts from Python
app.post('/api/alert', (req, res) => {
    const alertData = req.body;
    console.log('[Backend] Received Alert:', alertData);

    const newAlert = {
        ...alertData,
        timestamp: new Date().toISOString()
    };

    latestAlert = newAlert;
    alertHistory.unshift(newAlert); // Add to beginning
    if (alertHistory.length > 1000) alertHistory.pop(); // Limit to 1000

    res.status(200).send('Alert received');
});

// API Endpoint to receive status updates from Python and return command
app.post('/api/status', (req, res) => {
    const statusData = req.body;

    systemStatus = {
        ...systemStatus,
        ...statusData,
        lastHeartbeat: new Date().toISOString()
    };

    // Return the current command to the Python client
    // Only send command once, then reset to null to avoid repeated execution
    const commandToSend = systemCommand;
    if (systemCommand) {
        systemCommand = null;
    }

    res.json({ command: commandToSend });
});

// API Endpoint for Frontend to control system
app.post('/api/control', (req, res) => {
    const { command } = req.body;
    if (command === 'start' || command === 'stop') {
        systemCommand = command;
        lastControlCommand = command;
        if (command === 'stop') {
            systemStatus = {
                ...systemStatus,
                radarConnected: false,
                lastHeartbeat: new Date().toISOString()
            };
        }
        console.log(`[Backend] System command set to: ${command}`);

        // Broadcast command to Python client via WebSocket
        const payload = JSON.stringify({
            type: 'CONTROL',
            action: command.toUpperCase()
        });

        wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(payload);
            }
        });

        res.json({ success: true, command: systemCommand });
    } else {
        res.status(400).json({ success: false, message: 'Invalid command' });
    }
});

// API Endpoint for Frontend to update settings
app.post('/api/settings', (req, res) => {
    const { speed, height } = req.body;
    console.log(`[Backend] Received settings update: Speed=${speed}, Height=${height}`);

    // Broadcast settings to Python client via WebSocket
    const payload = JSON.stringify({
        type: 'SETTINGS',
        data: {
            speed: speed,
            height: height
        }
    });

    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(payload);
        }
    });

    res.json({ success: true, message: 'Settings updated' });
});

// API Endpoint for Frontend to poll alert
app.get('/api/alert', (req, res) => {
    res.json(latestAlert);
});

// API Endpoint for Frontend to poll alert history
app.get('/api/alerts', (req, res) => {
    res.json(alertHistory);
});

// API Endpoint for Frontend to poll status
app.get('/api/status', (req, res) => {
    // Check if radar is actually connected (heartbeat within last 5 seconds)
    const now = new Date();
    const last = systemStatus.lastHeartbeat ? new Date(systemStatus.lastHeartbeat) : null;
    const isLive = last && (now - last < 5000);
    const realtimeFresh = lastRealtimeAt && (Date.now() - lastRealtimeAt < 5000);
    const radarConnected = lastControlCommand === 'stop'
        ? false
        : Boolean((isLive && systemStatus.radarConnected) || realtimeFresh);

    res.json({
        ...systemStatus,
        radarConnected,
        backendConnected: true,
        serverTime: new Date().toISOString(),
        websocketClients: wss.clients.size
    });
});

app.get('/api/realtime', (req, res) => {
    res.json(latestRealtime || {
        type: 'realtime_data',
        points: [],
        tracks: [],
        skeletons: [],
        vitals: [],
        meta: {},
        timestamp: null
    });
});

app.get('/api/blackbox/list', (req, res) => {
    fs.readdir(BLACKBOX_DIR, { withFileTypes: true }, (error, entries) => {
        if (error) {
            res.json([]);
            return;
        }
        const files = entries
            .filter((entry) => entry.isFile())
            .map((entry) => entry.name)
            .filter((name) => /\.(json|csv|bin|txt|log|npz)$/i.test(name))
            .sort()
            .reverse()
            .slice(0, 100);
        res.json(files);
    });
});

app.get('/api/blackbox/:name', (req, res) => {
    const fileName = path.basename(req.params.name || '');
    const basePath = path.resolve(BLACKBOX_DIR);
    const fullPath = path.resolve(basePath, fileName);
    if (!fileName || !fullPath.startsWith(basePath)) {
        res.status(400).send('Invalid file name');
        return;
    }
    res.sendFile(fullPath, (error) => {
        if (error) {
            res.status(404).send('File not found');
        }
    });
});

app.get('/health', (req, res) => {
    res.json({
        ok: true,
        serverTime: new Date().toISOString()
    });
});

if (fs.existsSync(FRONTEND_DIST)) {
    app.use(express.static(FRONTEND_DIST));
}

app.get('*', (req, res) => {
    if (req.path.startsWith('/api/')) {
        res.status(404).json({ error: 'Not found' });
        return;
    }

    const indexPath = path.join(FRONTEND_DIST, 'index.html');
    if (fs.existsSync(indexPath)) {
        res.sendFile(indexPath);
        return;
    }

    res.send('Fall Detection Backend is running');
});

server.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on http://0.0.0.0:${PORT}`);
});

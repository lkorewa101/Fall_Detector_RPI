import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webview_flutter/webview_flutter.dart';

const _prefDashboardUrl = 'dashboard_url';

const _bg = Color(0xFFEEF4F8);
const _panel = Color(0xFFFFFFFF);
const _ink = Color(0xFF172033);
const _muted = Color(0xFF718096);
const _line = Color(0xFFD7E0EA);
const _blue = Color(0xFF2868C7);
const _navy = Color(0xFF103A62);
const _danger = Color(0xFFD34A4D);

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const FallDashboardMobileApp());
}

class FallDashboardMobileApp extends StatelessWidget {
  const FallDashboardMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '00 병원 낙상 감지 시스템',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: _bg,
        colorScheme: ColorScheme.fromSeed(
          seedColor: _blue,
          brightness: Brightness.light,
          primary: _blue,
          surface: _panel,
          error: _danger,
        ),
        appBarTheme: const AppBarTheme(
          centerTitle: false,
          backgroundColor: _panel,
          foregroundColor: _ink,
          surfaceTintColor: Colors.transparent,
          elevation: 0,
        ),
      ),
      home: const DashboardShell(),
    );
  }
}

class DashboardShell extends StatefulWidget {
  const DashboardShell({super.key});

  @override
  State<DashboardShell> createState() => _DashboardShellState();
}

class _DashboardShellState extends State<DashboardShell> {
  final _urlController = TextEditingController();
  WebViewController? _webController;
  String? _currentUrl;
  String? _loadError;
  bool _loadingPreference = true;
  int _progress = 0;

  @override
  void initState() {
    super.initState();
    _loadSavedUrl();
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _loadSavedUrl() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_prefDashboardUrl) ?? '';
    if (!mounted) return;

    _urlController.text = saved;
    setState(() {
      _loadingPreference = false;
    });

    if (saved.isNotEmpty) {
      await _connect(saved);
    }
  }

  Future<void> _connect(String rawUrl) async {
    final url = _normalizeDashboardUrl(rawUrl);
    final uri = Uri.tryParse(url);
    if (uri == null || !uri.hasScheme || uri.host.isEmpty) {
      setState(() {
        _loadError = '주소를 확인해 주세요.';
      });
      return;
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefDashboardUrl, url);

    final controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(_panel)
      ..setNavigationDelegate(
        NavigationDelegate(
          onProgress: (value) {
            if (mounted) setState(() => _progress = value);
          },
          onPageStarted: (_) {
            if (mounted) {
              setState(() {
                _progress = 0;
                _loadError = null;
              });
            }
          },
          onPageFinished: (_) {
            if (mounted) setState(() => _progress = 100);
          },
          onWebResourceError: (error) {
            if (!mounted || error.isForMainFrame != true) return;
            setState(() {
              _loadError = '대시보드에 연결할 수 없습니다. 터널 주소와 웹 서버 실행 상태를 확인해 주세요.';
            });
          },
        ),
      );

    setState(() {
      _webController = controller;
      _currentUrl = url;
      _loadError = null;
      _progress = 0;
      _urlController.text = url;
    });

    await controller.loadRequest(uri);
  }

  String _normalizeDashboardUrl(String rawValue) {
    final trimmed = rawValue.trim();
    if (trimmed.isEmpty) return '';
    if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
      return trimmed;
    }
    if (trimmed.contains('trycloudflare.com') || trimmed.contains('ngrok')) {
      return 'https://$trimmed';
    }
    return 'http://$trimmed';
  }

  Future<void> _showAddressDialog() async {
    final editingController = TextEditingController(
      text: _currentUrl ?? _urlController.text,
    );
    final nextUrl = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('대시보드 주소'),
        content: TextField(
          controller: editingController,
          autofocus: true,
          keyboardType: TextInputType.url,
          decoration: const InputDecoration(
            hintText: 'https://xxxx.trycloudflare.com',
            labelText: 'Cloudflare/ngrok 주소',
          ),
          onSubmitted: (value) => Navigator.of(context).pop(value),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('취소'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(editingController.text),
            child: const Text('연결'),
          ),
        ],
      ),
    );
    editingController.dispose();
    if (nextUrl == null) return;
    await _connect(nextUrl);
  }

  @override
  Widget build(BuildContext context) {
    if (_loadingPreference) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        titleSpacing: 14,
        title: Row(
          children: [
            const _LogoMark(),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    '00 병원 낙상 감지 시스템',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w900),
                  ),
                  Text(
                    _currentUrl == null
                        ? '외부 대시보드 연결 대기'
                        : Uri.parse(_currentUrl!).host,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: _muted, fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          if (_webController != null)
            IconButton(
              tooltip: '새로고침',
              onPressed: () => _webController?.reload(),
              icon: const Icon(Icons.refresh_rounded),
            ),
          IconButton(
            tooltip: '주소 변경',
            onPressed: _showAddressDialog,
            icon: const Icon(Icons.settings_rounded),
          ),
        ],
      ),
      body: _webController == null
          ? _SetupView(controller: _urlController, onConnect: _connect)
          : _webViewBody(),
    );
  }

  Widget _webViewBody() {
    return Stack(
      children: [
        WebViewWidget(controller: _webController!),
        if (_progress > 0 && _progress < 100)
          LinearProgressIndicator(
            value: _progress / 100,
            minHeight: 3,
            backgroundColor: Colors.transparent,
          ),
        if (_loadError != null)
          Positioned(
            left: 12,
            right: 12,
            bottom: 12,
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: _danger,
                borderRadius: BorderRadius.circular(8),
                boxShadow: const [
                  BoxShadow(color: Color(0x22000000), blurRadius: 18),
                ],
              ),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text(
                  _loadError!,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class _SetupView extends StatelessWidget {
  const _SetupView({required this.controller, required this.onConnect});

  final TextEditingController controller;
  final ValueChanged<String> onConnect;

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(18),
      children: [
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: _panel,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: _line),
            boxShadow: const [
              BoxShadow(
                color: Color(0x11172033),
                blurRadius: 24,
                offset: Offset(0, 12),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                '외부 대시보드 연결',
                style: TextStyle(
                  color: _ink,
                  fontSize: 25,
                  fontWeight: FontWeight.w900,
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'PC에서 Cloudflare Tunnel 또는 ngrok을 실행한 뒤 생성된 HTTPS 주소를 입력하세요.',
                style: TextStyle(color: _muted, height: 1.45),
              ),
              const SizedBox(height: 18),
              TextField(
                controller: controller,
                keyboardType: TextInputType.url,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: '대시보드 주소',
                  hintText: 'https://xxxx.trycloudflare.com',
                ),
                onSubmitted: onConnect,
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                height: 48,
                child: FilledButton.icon(
                  onPressed: () => onConnect(controller.text),
                  icon: const Icon(Icons.open_in_new_rounded),
                  label: const Text('대시보드 열기'),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 14),
        const _InstructionCard(),
      ],
    );
  }
}

class _InstructionCard extends StatelessWidget {
  const _InstructionCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFE8F1FF),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFFC9DAF7)),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'PC에서 실행 순서',
            style: TextStyle(
              color: _navy,
              fontSize: 16,
              fontWeight: FontWeight.w900,
            ),
          ),
          SizedBox(height: 10),
          _StepText('1. 웹/데스크탑 앱을 실행합니다.'),
          _StepText('2. cloudflared tunnel --url http://localhost:8000'),
          _StepText('3. 출력된 https://...trycloudflare.com 주소를 앱에 입력합니다.'),
        ],
      ),
    );
  }
}

class _StepText extends StatelessWidget {
  const _StepText(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 7),
      child: Text(text, style: const TextStyle(color: _ink, height: 1.35)),
    );
  }
}

class _LogoMark extends StatelessWidget {
  const _LogoMark();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 36,
      height: 36,
      decoration: BoxDecoration(
        color: _navy,
        borderRadius: BorderRadius.circular(9),
      ),
      child: const Icon(
        Icons.local_hospital_rounded,
        color: Colors.white,
        size: 24,
      ),
    );
  }
}

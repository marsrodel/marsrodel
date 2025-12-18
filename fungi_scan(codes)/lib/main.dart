import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:camera/camera.dart';
import 'dart:math' as math;
import 'dart:ui';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:fl_chart/fl_chart.dart';



Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(const MainApp());
}

List<double> softmax(List<double> x) {
  if (x.isEmpty) return x;
  final maxVal = x.reduce((a, b) => a > b ? a : b);
  final exps = x.map((v) => math.exp(v - maxVal)).toList();
  final sum = exps.fold<double>(0, (s, v) => s + v);
  if (sum == 0) {
    return List<double>.filled(x.length, 1.0 / x.length);
  }
  return exps.map((v) => v / sum).toList();
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fungi Variety',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primaryColor: Colors.red, useMaterial3: true),
      home: const HomePage(),
    );
  }
}

class PolkaDotPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final random = math.Random(98765); // New seed for wider distribution
    final paint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;

    // Divide the space into sections for better distribution
    final sections = 5;
    final sectionWidth = size.width / sections;

    // Create different sized dots for more natural look
    final dots = [
      {
        "size": 14.0,
        "count": 3,
        "opacity": 0.9,
      }, // Slightly smaller but more spread
      {"size": 10.0, "count": 4, "opacity": 0.85},
      {"size": 8.0, "count": 5, "opacity": 0.8},
    ];

    // Function to check if a point is too close to existing dots
    List<Offset> existingDots = [];
    bool isTooClose(Offset newPoint, double minDistance) {
      for (final dot in existingDots) {
        if ((dot - newPoint).distance < minDistance) {
          return true;
        }
      }
      return false;
    }

    // Draw dots section by section
    for (int section = 0; section < sections; section++) {
      for (final dot in dots) {
        paint.color = Colors.white.withOpacity(dot["opacity"] as double);
        final dotSize = dot["size"] as double;

        for (int i = 0; i < (dot["count"] as int); i++) {
          // Try to place dot up to 10 times to ensure good spacing
          for (int attempt = 0; attempt < 10; attempt++) {
            final x =
                section * sectionWidth + random.nextDouble() * sectionWidth;
            final y = random.nextDouble() * size.height;
            final newPoint = Offset(x, y);

            // Check if dot is well-spaced and within bounds
            if (!isTooClose(newPoint, dotSize * 2) &&
                x > dotSize &&
                x < size.width - dotSize &&
                y > dotSize &&
                y < size.height - dotSize) {
              // Draw glow/outline effect
              paint.maskFilter = const MaskFilter.blur(BlurStyle.outer, 2);
              paint.color = Colors.white.withOpacity(0.3);
              canvas.drawCircle(newPoint, (dotSize / 2) + 1, paint);

              // Draw main dot
              paint.maskFilter = null;
              paint.color = Colors.white.withOpacity(dot["opacity"] as double);
              canvas.drawCircle(newPoint, dotSize / 2, paint);

              existingDots.add(newPoint);
              break;
            }
          }
        }
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}

class Classifier {
  tfl.Interpreter? _interpreter;
  List<String> _labels = [];
  int _inputSize = 224;
  bool _isFloat = true;

  Future<void> load() async {
    if (_interpreter != null) return;
    final interpreter = await tfl.Interpreter.fromAsset('assets/model_unquant.tflite');
    _interpreter = interpreter;
    final inputShape = interpreter.getInputTensor(0).shape;
    _inputSize = inputShape.length >= 3 ? inputShape[1] : 224;
    _isFloat = true;
    final labelsStr = await rootBundle.loadString('assets/labels.txt');
    _labels = labelsStr
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .map((e) => e.replaceFirst(RegExp(r'^\s*\d+\s*'), '').trim())
        .toList();
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }

  Future<List<double>> classifyProbs(File file) async {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('Interpreter not loaded');
    }
    final bytes = await file.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) throw StateError('Failed to decode image');
    // Center-crop to square to avoid distortion, then resize to model input
    final shortest = decoded.width < decoded.height ? decoded.width : decoded.height;
    final cropX = ((decoded.width - shortest) / 2).floor();
    final cropY = ((decoded.height - shortest) / 2).floor();
    final square = img.copyCrop(decoded, x: cropX, y: cropY, width: shortest, height: shortest);
    final resized = img.copyResize(square, width: _inputSize, height: _inputSize);
    if (_isFloat) {
      final input = List.generate(1, (_) => List.generate(_inputSize, (_) => List.generate(_inputSize, (_) => List.filled(3, 0.0))));
      final rgba = resized.getBytes(order: img.ChannelOrder.rgba);
      for (int y = 0; y < _inputSize; y++) {
        for (int x = 0; x < _inputSize; x++) {
          final base = (y * _inputSize + x) * 4;
          // Normalize to [0, 1]
          final r = rgba[base] / 255.0;
          final g = rgba[base + 1] / 255.0;
          final b = rgba[base + 2] / 255.0;
          input[0][y][x][0] = r;
          input[0][y][x][1] = g;
          input[0][y][x][2] = b;
        }
      }
      final outputTensor = interpreter.getOutputTensor(0);
      final numClasses = outputTensor.shape.last;
      final output = [List.filled(numClasses, 0.0)];
      interpreter.run(input, output);
      final probs = (output[0] as List).cast<double>();
      return probs;
    } else {
      final input = List.generate(1, (_) => List.generate(_inputSize, (_) => List.generate(_inputSize, (_) => List.filled(3, 0))));
      final rgba = resized.getBytes(order: img.ChannelOrder.rgba);
      for (int y = 0; y < _inputSize; y++) {
        for (int x = 0; x < _inputSize; x++) {
          final base = (y * _inputSize + x) * 4;
          input[0][y][x][0] = rgba[base];
          input[0][y][x][1] = rgba[base + 1];
          input[0][y][x][2] = rgba[base + 2];
        }
      }
      final outputTensor = interpreter.getOutputTensor(0);
      final numClasses = outputTensor.shape.last;
      final output = [List.filled(numClasses, 0)];
      interpreter.run(input, output);
      final probs = (output[0] as List).map((e) => (e as int).toDouble()).toList();
      return probs;
    }
  }

  Future<List<Map<String, dynamic>>> classify(File file, {int topK = 3}) async {
    final probs = await classifyProbs(file);
    final total = probs.fold<double>(0, (s, v) => s + v);
    final norm = total == 0 ? probs : probs.map((v) => v / total).toList();
    final results = <Map<String, dynamic>>[];
    for (int i = 0; i < norm.length; i++) {
      final label = i < _labels.length ? _labels[i] : 'Class $i';
      results.add({'label': label, 'index': i, 'confidence': norm[i]});
    }
    results.sort((a, b) => (b['confidence'] as double).compareTo(a['confidence'] as double));
    return results.take(topK).toList();
  }

  String formatTopResult(List<Map<String, dynamic>> results) {
    if (results.isEmpty) return 'No result';
    final best = results.first;
    return '${best['label']}';
  }
}

class FungiInfo {
  final String name;
  final String description;
  final String imagePath;

  const FungiInfo({
    required this.name,
    required this.description,
    required this.imagePath,
  });
}

const List<FungiInfo> kFungiDictionary = [
  FungiInfo(
    name: 'Button Mushroom',
    description:
        'A small, white mushroom that is commonly used in cooking. It has a mild flavor and soft texture. This is the type you often see in grocery stores.',
    imagePath: 'assets/photos/button.jpg',
  ),
  FungiInfo(
    name: 'Oyster Mushroom',
    description:
        'A mushroom with wide, fan-shaped caps that look like oysters. It has a soft, delicate texture and a slightly sweet, mild taste. Often used in stir-fries and soups.',
    imagePath: 'assets/photos/oyster.jpg',
  ),
  FungiInfo(
    name: 'Enoki Mushroom',
    description:
        'A mushroom with long, thin stems and tiny white caps. It grows in tight bunches and has a crunchy texture. Common in ramen, hotpot, and salads.',
    imagePath: 'assets/photos/enoki.jpg',
  ),
  FungiInfo(
    name: 'Morel Mushroom',
    description:
        'A rare mushroom with a honeycomb-like cap full of holes. It has a rich, earthy flavor and is considered a gourmet ingredient.',
    imagePath: 'assets/photos/morel.jpg',
  ),
  FungiInfo(
    name: 'Chanterelle Mushroom',
    description:
        'A bright yellow or orange mushroom shaped like a small trumpet. It has a fruity smell and a slightly peppery taste. Popular in fine dining dishes.',
    imagePath: 'assets/photos/chanterelles.jpg',
  ),
  FungiInfo(
    name: 'Black Trumpet Mushroom',
    description:
        'A dark, funnel-shaped mushroom that almost looks like a hollow trumpet. It has a smoky, deep flavor and is often used in sauces.',
    imagePath: 'assets/photos/black_trumpet.jpg',
  ),
  FungiInfo(
    name: 'Fly Agaric Mushroom',
    description:
        'A bright red mushroom with white spots. It is famous in fairy tales and video games. Not safe to eat, as it can be poisonous.',
    imagePath: 'assets/photos/fly_agaric.jpg',
  ),
  FungiInfo(
    name: 'Reishi Mushroom',
    description:
        'A tough, woody mushroom often used in traditional medicine. It has a shiny, reddish surface and is usually made into teas or supplements, not eaten as food.',
    imagePath: 'assets/photos/reishi.jpg',
  ),
  FungiInfo(
    name: 'Coral Fungus',
    description:
        'A fungus that looks like underwater coral, with many branching arms. It comes in different colors and grows on the forest floor.',
    imagePath: 'assets/photos/coral.jpg',
  ),
  FungiInfo(
    name: 'Bleeding Tooth Fungus',
    description:
        'A white fungus that “bleeds” bright red liquid droplets. It looks unusual and is not edible. The red appearance comes from natural pigments.',
    imagePath: 'assets/photos/bleeding_tooth.jpeg',
  ),
];

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();
  bool _picking = false; // upload photo
  bool _scanning = false; // scan via camera
  final Classifier _classifier = Classifier();
  bool _showDictionary = false;
  int _uploadCountdown = 0;

  Future<void> _storeLog(String className, double accuracyPercent) async {
    try {
      final docRef = FirebaseFirestore.instance
          .collection('Maraon-FungiVariety')
          .doc('Maraon_FungiVariety_Logs');

      await docRef.set({
        'logs': FieldValue.arrayUnion([
          {
            'ClassType': className,
            'Accuracy_Rate': accuracyPercent,
            'Time': Timestamp.now(),
          },
        ]),
      }, SetOptions(merge: true));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to store log: $e')),
      );
    }
  }

  Future<void> _runClassificationAndShow(File file) async {
    await _classifier.load();
    final probs = await _classifier.classifyProbs(file);
    final results = <Map<String, dynamic>>[];
    for (int i = 0; i < probs.length; i++) {
      results.add({'label': i < _classifier._labels.length ? _classifier._labels[i] : 'Class $i', 'index': i, 'confidence': probs[i]});
    }
    results.sort((a, b) => (b['confidence'] as double).compareTo(a['confidence'] as double));
    if (!mounted) return;
    showDialog(
      context: context,
      builder: (context) {
        // Known classes (show all of them, even if probability is 0)
        const classNames = [
          'Button Mushroom',
          'Oyster Mushroom',
          'Enoki Mushroom',
          'Morel Mushroom',
          'Chanterelle Mushroom',
          'Black Trumpet Mushroom',
          'Fly Agaric Mushroom',
          'Reishi Mushroom',
          'Coral Fungus',
          'Bleeding Tooth Fungus',
        ];

        // Map from label to confidence
        final Map<String, double> confidenceByLabel = {};
        for (final r in results) {
          final label = r['label'] as String;
          final conf = (r['confidence'] as num).toDouble();
          confidenceByLabel[label] = conf;
        }

        return AlertDialog(
          title: Text(
            'Result:',
            style: GoogleFonts.poppins(fontWeight: FontWeight.w700),
            textAlign: TextAlign.center,
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                _classifier.formatTopResult(results),
                style: GoogleFonts.titanOne(
                  fontSize: 24,
                  color: Theme.of(context).primaryColor,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              Text(
                'Prediction distribution',
                style: GoogleFonts.poppins(fontSize: 16),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              SizedBox(
                height: 160,
                width: double.infinity,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: classNames.map((fullLabel) {
                    final shortLabel = fullLabel
                        .replaceAll(' Mushroom', '')
                        .replaceAll(' Fungus', '');
                    final percent = (confidenceByLabel[fullLabel] ?? 0.0) * 100.0;
                    final percentText = percent == 0
                        ? '0'
                        : percent.toStringAsFixed(2);
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 1),
                      child: Row(
                        children: [
                          SizedBox(
                            width: 70,
                            child: Text(
                              shortLabel,
                              style: GoogleFonts.poppins(fontSize: 10),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          const SizedBox(width: 6),
                          Expanded(
                            child: LayoutBuilder(
                              builder: (context, constraints) {
                                final maxWidth = constraints.maxWidth;
                                final barWidth = (percent.clamp(0, 100) / 100.0) * maxWidth;
                                return Stack(
                                  children: [
                                    Container(
                                      height: 6,
                                      decoration: BoxDecoration(
                                        color: Colors.grey.shade300,
                                        borderRadius: BorderRadius.circular(3),
                                      ),
                                    ),
                                    Container(
                                      height: 6,
                                      width: barWidth,
                                      decoration: BoxDecoration(
                                        color: Theme.of(context).primaryColor,
                                        borderRadius: BorderRadius.circular(3),
                                      ),
                                    ),
                                  ],
                                );
                              },
                            ),
                          ),
                          const SizedBox(width: 6),
                          SizedBox(
                            width: 40,
                            child: Text(
                              percentText,
                              style: GoogleFonts.poppins(fontSize: 10),
                              textAlign: TextAlign.right,
                            ),
                          ),
                        ],
                      ),
                    );
                  }).toList(),
                ),
              ),
            ],
          ),
          actionsAlignment: MainAxisAlignment.center,
          actions: [
            ElevatedButton(
              onPressed: () async {
                final best = results.isNotEmpty ? results.first : null;
                if (best != null) {
                  final label = best['label'] as String;
                  final confidence = (best['confidence'] as num).toDouble();
                  await _storeLog(label, confidence * 100);
                }
                if (mounted) {
                  Navigator.of(context).pop();
                }
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).primaryColor,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
              child: Text('Store', style: GoogleFonts.poppins()),
            ),
            const SizedBox(width: 12),
            OutlinedButton(
              onPressed: () => Navigator.of(context).pop(),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.black,
                side: const BorderSide(color: Colors.black),
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
              child: Text('Cancel', style: GoogleFonts.poppins()),
            ),
          ],
        );
      },
    );
  }

  Future<void> _scanWithSystemCamera() async {
    if (_scanning || !mounted) return;
    setState(() {
      _scanning = true;
    });
    try {
      final picked = await _picker.pickImage(source: ImageSource.camera);
      if (!mounted) return;
      if (picked == null) {
        return;
      }
      await _runClassificationAndShow(File(picked.path));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to scan: $e')),
      );
    } finally {
      if (!mounted) return;
      setState(() {
        _scanning = false;
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _classifier.load();
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    if (_picking) {
      return;
    }
    setState(() {
      _picking = true;
    });
    try {
      final picked = await _picker.pickImage(source: ImageSource.gallery);
      if (!mounted) {
        return;
      }
      if (picked != null) {
        await _runClassificationAndShow(File(picked.path));
      }
    } catch (e) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to pick image: $e'),
          duration: const Duration(seconds: 3),
        ),
      );
    } finally {
      if (mounted) {
        setState(() {
          _picking = false;
          _uploadCountdown = 0;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Stack(
          alignment: Alignment.center,
          children: [
            Text(
              'Fungi Scan',
              style: GoogleFonts.titanOne(
                fontSize: 40,
                foreground: Paint()
                  ..style = PaintingStyle.stroke
                  ..strokeWidth = 5
                  ..color = Colors.black87,
              ),
            ),
            Text(
              'Fungi Scan',
              style: GoogleFonts.titanOne(
                fontSize: 40,
                color: Colors.white,
                shadows: [
                  Shadow(
                    offset: Offset(0, 0),
                    blurRadius: 3,
                    color: Colors.black87,
                  ),
                  Shadow(
                    offset: Offset(2, 0),
                    blurRadius: 3,
                    color: Colors.black54,
                  ),
                  Shadow(
                    offset: Offset(-2, 0),
                    blurRadius: 3,
                    color: Colors.black54,
                  ),
                  Shadow(
                    offset: Offset(0, 2),
                    blurRadius: 3,
                    color: Colors.black54,
                  ),
                  Shadow(
                    offset: Offset(0, -2),
                    blurRadius: 3,
                    color: Colors.black54,
                  ),
                ],
              ),
            ),
          ],
        ),
        centerTitle: true,
        flexibleSpace: Stack(
          children: [
            Container(color: Colors.red),
            CustomPaint(painter: PolkaDotPainter(), child: Container()),
          ],
        ),
        backgroundColor: Colors.red,
        foregroundColor: Colors.white,
      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: ImageFiltered(
              imageFilter: ImageFilter.blur(sigmaX: 2, sigmaY: 1),
              child: Image.asset(
                'assets/background.jpeg',
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) {
                  return Container(
                    color: Colors.grey[200],
                    child: Center(
                      child: Icon(
                        Icons.image_not_supported,
                        size: 64,
                        color: Colors.grey[400],
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.topCenter,
              child: SingleChildScrollView(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(24, 16, 24, 0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      FractionallySizedBox(
                        widthFactor: 1,
                        child: Container(
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.85),
                            borderRadius: BorderRadius.circular(28),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.15),
                                blurRadius: 16,
                                offset: const Offset(0, 8),
                              ),
                            ],
                          ),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 24,
                            vertical: 32,
                          ),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Container(
                                width: 72,
                                height: 72,
                                decoration: BoxDecoration(
                                  color: Theme.of(context)
                                      .primaryColor
                                      .withOpacity(0.15),
                                  shape: BoxShape.circle,
                                ),
                                child: Center(
                                  child: Text(
                                    '🍄',
                                    style: TextStyle(
                                      fontSize: 40,
                                      color: Theme.of(context).primaryColor,
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(height: 20),
                              Text(
                                'Fungi Scan',
                                style: GoogleFonts.titanOne(
                                  fontSize: 32,
                                  color: Theme.of(context).primaryColor,
                                ),
                                textAlign: TextAlign.center,
                              ),
                              const SizedBox(height: 12),
                              Text(
                                'Scan or upload photos to see which fungi species you have found.',
                                style: GoogleFonts.poppins(
                                  fontSize: 16,
                                  color: Colors.black87,
                                  height: 1.4,
                                ),
                                textAlign: TextAlign.center,
                              ),
                              const SizedBox(height: 24),
                              SizedBox(
                                width: double.infinity,
                                child: ElevatedButton.icon(
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor:
                                        Theme.of(context).primaryColor,
                                    foregroundColor: Colors.white,
                                    padding: const EdgeInsets.symmetric(
                                      vertical: 18,
                                      horizontal: 16,
                                    ),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(20),
                                    ),
                                  ),
                                  onPressed: _scanWithSystemCamera,
                                  icon:
                                      const Icon(Icons.camera_alt_outlined),
                                  label: Text(
                                    'Scan Fungus',
                                    style: GoogleFonts.poppins(
                                      fontSize: 18,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(height: 12),
                              SizedBox(
                                width: double.infinity,
                                child: OutlinedButton.icon(
                                  style: OutlinedButton.styleFrom(
                                    foregroundColor:
                                        Theme.of(context).primaryColor,
                                    side: BorderSide(
                                      color: Theme.of(context).primaryColor,
                                      width: 2,
                                    ),
                                    padding: const EdgeInsets.symmetric(
                                      vertical: 18,
                                      horizontal: 16,
                                    ),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(20),
                                    ),
                                  ),
                                  onPressed: _picking ? null : _pickImage,
                                  icon: const Icon(Icons.upload_file_outlined),
                                  label: _picking
                                      ? SizedBox(
                                          height: 20,
                                          width: 20,
                                          child: Stack(
                                            alignment: Alignment.center,
                                            children: [
                                              CircularProgressIndicator(
                                                strokeWidth: 2,
                                                valueColor: AlwaysStoppedAnimation(
                                                  Theme.of(context).primaryColor,
                                                ),
                                              ),
                                              if (_uploadCountdown > 0)
                                                Text(
                                                  '$_uploadCountdown',
                                                  style: GoogleFonts.poppins(
                                                    fontSize: 11,
                                                    fontWeight: FontWeight.w600,
                                                    color: Theme.of(context).primaryColor,
                                                  ),
                                                ),
                                            ],
                                          ),
                                        )
                                      : Text(
                                          'Upload Photo',
                                          style: GoogleFonts.poppins(
                                            fontSize: 18,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.9),
                          borderRadius: BorderRadius.circular(24),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.1),
                              blurRadius: 12,
                              offset: const Offset(0, 6),
                            ),
                          ],
                        ),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 24,
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            InkWell(
                              borderRadius: BorderRadius.circular(16),
                              onTap: () {
                                setState(() {
                                  _showDictionary = !_showDictionary;
                                });
                              },
                              child: Row(
                                children: [
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          'Fungi Dictionary',
                                          style: GoogleFonts.titanOne(
                                            fontSize: 24,
                                            color:
                                                Theme.of(context).primaryColor,
                                          ),
                                        ),
                                        const SizedBox(height: 4),
                                        Text(
                                          'Tap to ${_showDictionary ? 'hide' : 'view'} the list of fungi classes.',
                                          style: GoogleFonts.poppins(
                                            fontSize: 14,
                                            color: Colors.black87,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  Icon(
                                    _showDictionary
                                        ? Icons.keyboard_arrow_up
                                        : Icons.keyboard_arrow_down,
                                    color: Colors.black54,
                                  ),
                                ],
                              ),
                            ),
                            if (_showDictionary) ...[
                              const SizedBox(height: 16),
                              ...kFungiDictionary.map(
                                (fungus) => Padding(
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 4),
                                  child: InkWell(
                                    borderRadius: BorderRadius.circular(16),
                                    onTap: () {
                                      showDialog(
                                        context: context,
                                        builder: (context) {
                                          return AlertDialog(
                                            title: Text(
                                              fungus.name,
                                              style: GoogleFonts.titanOne(
                                                fontSize: 22,
                                                color: Theme.of(context)
                                                    .primaryColor,
                                              ),
                                            ),
                                            content: Column(
                                              mainAxisSize: MainAxisSize.min,
                                              crossAxisAlignment:
                                                  CrossAxisAlignment.start,
                                              children: [
                                                ClipRRect(
                                                  borderRadius:
                                                      BorderRadius.circular(16),
                                                  child: AspectRatio(
                                                    aspectRatio: 1,
                                                    child: Image.asset(
                                                      fungus.imagePath,
                                                      fit: BoxFit.cover,
                                                    ),
                                                  ),
                                                ),
                                                const SizedBox(height: 12),
                                                Text(
                                                  fungus.description,
                                                  style: GoogleFonts.poppins(
                                                    fontSize: 14,
                                                    color: Colors.black87,
                                                    height: 1.4,
                                                  ),
                                                ),
                                              ],
                                            ),
                                          );
                                        },
                                      );
                                    },
                                    child: Row(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.center,
                                      children: [
                                        ClipRRect(
                                          borderRadius:
                                              BorderRadius.circular(16),
                                          child: Image.asset(
                                            fungus.imagePath,
                                            height: 48,
                                            width: 48,
                                            fit: BoxFit.cover,
                                          ),
                                        ),
                                        const SizedBox(width: 12),
                                        Expanded(
                                          child: Text(
                                            fungus.name,
                                            style: GoogleFonts.poppins(
                                              fontSize: 16,
                                              fontWeight: FontWeight.w600,
                                            ),
                                          ),
                                        ),
                                        const Icon(
                                          Icons.chevron_right,
                                          color: Colors.black45,
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Theme.of(context).primaryColor,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                              vertical: 18,
                              horizontal: 16,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                          onPressed: () {
                            Navigator.of(context).push(
                              MaterialPageRoute(
                                builder: (_) => const AnalyticsPage(),
                              ),
                            );
                          },
                          icon: const Icon(Icons.bar_chart_rounded, color: Colors.white),
                          label: Text(
                            'Analytics',
                            style: GoogleFonts.poppins(
                              fontSize: 18,
                              fontWeight: FontWeight.w600,
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 40),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  bool _permissionDenied = false;
  bool _initializing = false;
  bool _noAvailableCamera = false;
  final Classifier _classifier = Classifier();
  bool _processing = false;
  int _countdown = 0;
  bool _flashOn = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
    _classifier.load();
  }

  Future<void> _initializeCamera() async {
    if (_initializing) {
      return;
    }
    setState(() {
      _initializing = true;
      _permissionDenied = false;
      _noAvailableCamera = false;
    });
    try {
      final cameras = await availableCameras();
      if (!mounted) {
        return;
      }
      if (cameras.isEmpty) {
        setState(() {
          _noAvailableCamera = true;
        });
        return;
      }
      final selectedCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        selectedCamera,
        ResolutionPreset.max,
        enableAudio: false,
      );
      await _controller?.dispose();
      final initializeFuture = controller.initialize();
      setState(() {
        _controller = controller;
        _initializeControllerFuture = initializeFuture;
      });
      await initializeFuture;
      // Default flash off so it doesn't fire unexpectedly
      await controller.setFlashMode(FlashMode.off);
    } on CameraException catch (e) {
      if (!mounted) {
        return;
      }
      if (e.code == 'CameraAccessDenied' ||
          e.code == 'CameraAccessDeniedWithoutPrompt') {
        setState(() {
          _permissionDenied = true;
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _initializing = false;
        });
      }
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }
    if (state == AppLifecycleState.inactive) {
      controller.dispose();
      setState(() {
        _controller = null;
        _initializeControllerFuture = null;
      });
    } else if (state == AppLifecycleState.resumed && _controller == null) {
      _initializeCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _classifier.dispose();
    super.dispose();
  }

  Widget _buildCameraPreview() {
    if (_permissionDenied) {
      return const Center(
        child: Text(
          'Camera permission denied',
          style: TextStyle(color: Colors.white, fontSize: 18),
        ),
      );
    }
    if (_noAvailableCamera) {
      return const Center(
        child: Text(
          'No camera available',
          style: TextStyle(color: Colors.white, fontSize: 18),
        ),
      );
    }
    final controller = _controller;
    final future = _initializeControllerFuture;
    if (controller == null || future == null) {
      return const Center(child: CircularProgressIndicator());
    }
    return FutureBuilder<void>(
      future: future,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          return LayoutBuilder(
            builder: (context, constraints) {
              final size = math.min(
                constraints.maxWidth,
                constraints.maxHeight,
              );
              return Center(
                child: SizedBox(
                  width: size,
                  height: size,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(24),
                    child: FittedBox(
                      fit: BoxFit.cover,
                      child: SizedBox(
                        width: controller.value.previewSize?.height ?? size,
                        height: controller.value.previewSize?.width ?? size,
                        child: CameraPreview(controller),
                      ),
                    ),
                  ),
                ),
              );
            },
          );
        }
        if (snapshot.hasError) {
          return const Center(
            child: Text(
              'Failed to start camera',
              style: TextStyle(color: Colors.white, fontSize: 18),
            ),
          );
        }
        return const Center(child: CircularProgressIndicator());
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          Positioned.fill(
            child: Stack(
              children: [
                Positioned.fill(child: _buildCameraPreview()),
                Center(
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: Container(
                      margin: const EdgeInsets.all(32),
                      decoration: BoxDecoration(
                        color: Colors.transparent,
                        border: Border.all(
                          color: Colors.white.withOpacity(0.85),
                          width: 4,
                        ),
                        borderRadius: BorderRadius.circular(24),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.topCenter,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back, color: Colors.white),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                    IconButton(
                      iconSize: 26,
                      color: Colors.white,
                      onPressed: () async {
                        final controller = _controller;
                        if (controller == null || !controller.value.isInitialized) return;
                        final newOn = !_flashOn;
                        try {
                          await controller.setFlashMode(newOn ? FlashMode.torch : FlashMode.off);
                          if (mounted) {
                            setState(() {
                              _flashOn = newOn;
                            });
                          }
                        } catch (_) {}
                      },
                      icon: Icon(_flashOn ? Icons.flash_on : Icons.flash_off),
                    ),
                  ],
                ),
              ),
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: SafeArea(
              minimum: const EdgeInsets.only(bottom: 32),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (_processing)
                    Container(
                      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 32),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.06),
                        borderRadius: BorderRadius.circular(40),
                      ),
                      child: Center(
                        child: SizedBox(
                          height: 56,
                          width: 56,
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              CircularProgressIndicator(
                                strokeWidth: 4,
                                valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                              ),
                              Text(
                                '$_countdown',
                                style: GoogleFonts.poppins(
                                  fontSize: 20,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.white,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    )
                  else
                    SizedBox(
                      width: double.infinity,
                      child: Align(
                        alignment: Alignment.center,
                        child: ElevatedButton.icon(
                          onPressed: _scanAndClassifyFor10s,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Theme.of(context).primaryColor,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 18),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                          ),
                          icon: const Icon(Icons.camera),
                          label: Text(
                            'Scan & Identify',
                            style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// removed custom reshape extension; we build 2D lists directly for outputs

extension _CameraScanActions on _CameraPageState {
  Future<void> _scanAndClassifyFor10s() async {
    setState(() {
      _processing = true;
      _countdown = 0;
    });
    try {
      await _classifier.load();

      // Use system camera (native camera app)
      final picker = ImagePicker();
      final picked = await picker.pickImage(source: ImageSource.camera);
      if (!mounted) return;
      if (picked == null) {
        // User cancelled
        return;
      }

      final probs = await _classifier.classifyProbs(File(picked.path));
      final results = <Map<String, dynamic>>[];
      for (int i = 0; i < probs.length; i++) {
        results.add({'label': i < _classifier._labels.length ? _classifier._labels[i] : 'Class $i', 'index': i, 'confidence': probs[i]});
      }
      results.sort((a, b) => (b['confidence'] as double).compareTo(a['confidence'] as double));
      if (!mounted) return;
      final best = results.isNotEmpty ? results.first : null;

      // Same fixed class list as upload dialog
      const classNames = [
        'Button Mushroom',
        'Oyster Mushroom',
        'Enoki Mushroom',
        'Morel Mushroom',
        'Chanterelle Mushroom',
        'Black Trumpet Mushroom',
        'Fly Agaric Mushroom',
        'Reishi Mushroom',
        'Coral Fungus',
        'Bleeding Tooth Fungus',
      ];

      // Map label -> confidence
      final Map<String, double> confidenceByLabel = {};
      for (final r in results) {
        final label = r['label'] as String;
        final conf = (r['confidence'] as num).toDouble();
        confidenceByLabel[label] = conf;
      }

      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text(
            'Result:',
            style: GoogleFonts.poppins(fontWeight: FontWeight.w700),
            textAlign: TextAlign.center,
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              if (best != null)
                Text(
                  '${best['label']}',
                  style: GoogleFonts.titanOne(
                    fontSize: 24,
                    color: Theme.of(context).primaryColor,
                  ),
                  textAlign: TextAlign.center,
                ),
              const SizedBox(height: 12),
              Text(
                'Prediction distribution',
                style: GoogleFonts.poppins(fontSize: 16),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              SizedBox(
                height: 160,
                width: double.infinity,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: classNames.map((fullLabel) {
                    final shortLabel = fullLabel
                        .replaceAll(' Mushroom', '')
                        .replaceAll(' Fungus', '');
                    final percent = (confidenceByLabel[fullLabel] ?? 0.0) * 100.0;
                    final percentText = percent == 0
                        ? '0'
                        : percent.toStringAsFixed(2);
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 1),
                      child: Row(
                        children: [
                          SizedBox(
                            width: 70,
                            child: Text(
                              shortLabel,
                              style: GoogleFonts.poppins(fontSize: 10),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          const SizedBox(width: 6),
                          Expanded(
                            child: LayoutBuilder(
                              builder: (context, constraints) {
                                final maxWidth = constraints.maxWidth;
                                final barWidth = (percent.clamp(0, 100) / 100.0) * maxWidth;
                                return Stack(
                                  children: [
                                    Container(
                                      height: 6,
                                      decoration: BoxDecoration(
                                        color: Colors.grey.shade300,
                                        borderRadius: BorderRadius.circular(3),
                                      ),
                                    ),
                                    Container(
                                      height: 6,
                                      width: barWidth,
                                      decoration: BoxDecoration(
                                        color: Theme.of(context).primaryColor,
                                        borderRadius: BorderRadius.circular(3),
                                      ),
                                    ),
                                  ],
                                );
                              },
                            ),
                          ),
                          const SizedBox(width: 6),
                          SizedBox(
                            width: 40,
                            child: Text(
                              percentText,
                              style: GoogleFonts.poppins(fontSize: 10),
                              textAlign: TextAlign.right,
                            ),
                          ),
                        ],
                      ),
                    );
                  }).toList(),
                ),
              ),
            ],
          ),
          actionsAlignment: MainAxisAlignment.center,
          actions: [
            ElevatedButton(
              onPressed: () async {
                if (best != null) {
                  final label = best['label'] as String;
                  final confidence = (best['confidence'] as num).toDouble();
                  // Use HomePage's helper via context
                  final state = context.findAncestorStateOfType<_HomePageState>();
                  if (state != null) {
                    await state._storeLog(label, confidence * 100);
                  }
                }
                if (context.mounted) {
                  Navigator.of(context).pop();
                }
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).primaryColor,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
              child: Text('Store', style: GoogleFonts.poppins()),
            ),
            const SizedBox(width: 12),
            OutlinedButton(
              onPressed: () => Navigator.of(context).pop(),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.black,
                side: const BorderSide(color: Colors.black),
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
              child: Text('Cancel', style: GoogleFonts.poppins()),
            ),
          ],
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to scan: $e')));
    } finally {
      if (mounted) setState(() {
        _processing = false;
        _countdown = 0;
      });
    }
  }
  
}
class AnalyticsPage extends StatefulWidget {
  const AnalyticsPage({super.key});

  @override
  State<AnalyticsPage> createState() => _AnalyticsPageState();
}

class _AnalyticsPageState extends State<AnalyticsPage> {
  // Fetch logs from Firestore
  // Firestore structure from your screenshot:
  // Collection: Maraon-FungiVariety
  //   Document: Maraon_FungiVariety_Logs
  Stream<List<Map<String, dynamic>>> _getLogsStream() {
    return FirebaseFirestore.instance
        .collection('Maraon-FungiVariety')
        .doc('Maraon_FungiVariety_Logs')
        .snapshots()
        .map((snapshot) {
      final data = snapshot.data();
      if (data == null) return <Map<String, dynamic>>[];

      // Preferred structure: an array field 'logs', each item a map
      final dynamic rawLogs = data['logs'];
      if (rawLogs is List) {
        return rawLogs.map<Map<String, dynamic>>((item) {
          final Map<String, dynamic> log =
              item is Map<String, dynamic> ? item : Map<String, dynamic>.from(item as Map);

          String timeString = '';
          final dynamic rawTime = log['Time'];
          if (rawTime is Timestamp) {
            final dateTime = rawTime.toDate();
            timeString = '${dateTime.month}/${dateTime.day}/${dateTime.year} at ${dateTime.hour}:${dateTime.minute.toString().padLeft(2, '0')}';
          } else if (rawTime is String) {
            timeString = rawTime;
          }

          return {
            'classType': log['ClassType'] ?? '',
            'accuracyRate': (log['Accuracy_Rate'] ?? 0).toDouble(),
            'time': timeString,
            // Keep original DateTime for graphing when available
            'dateTime': rawTime is Timestamp ? rawTime.toDate() : null,
          };
        }).toList();
      }

      // Backwards compatibility: single log stored directly on the document
      String timeString = '';
      final dynamic rawTime = data['Time'];
      if (rawTime is Timestamp) {
        final dateTime = rawTime.toDate();
        timeString = '${dateTime.month}/${dateTime.day}/${dateTime.year} at ${dateTime.hour}:${dateTime.minute.toString().padLeft(2, '0')}';
      } else if (rawTime is String) {
        timeString = rawTime;
      }

      return [
        {
          'classType': data['ClassType'] ?? '',
          'accuracyRate': (data['Accuracy_Rate'] ?? 0).toDouble(),
          'time': timeString,
          'dateTime': rawTime is Timestamp ? rawTime.toDate() : null,
        },
      ];
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Analytics',
          style: GoogleFonts.titanOne(
            fontSize: 28,
            color: Colors.white,
          ),
        ),
        centerTitle: true,
        backgroundColor: Theme.of(context).primaryColor,
        foregroundColor: Colors.white,
      ),
      body: StreamBuilder<List<Map<String, dynamic>>>(
        stream: _getLogsStream(),
        builder: (context, snapshot) {
          // Show loading state
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          
          // Show error state
          if (snapshot.hasError) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(
                  'Error loading logs:\n${snapshot.error}',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.poppins(color: Colors.red),
                ),
              ),
            );
          }

          final logs = snapshot.data ?? [];

          // Prepare data for detections-per-class graph
          const classNames = [
            'Button Mushroom',
            'Oyster Mushroom',
            'Enoki Mushroom',
            'Morel Mushroom',
            'Chanterelle Mushroom',
            'Black Trumpet Mushroom',
            'Fly Agaric Mushroom',
            'Reishi Mushroom',
            'Coral Fungus',
            'Bleeding Tooth Fungus',
          ];

          final Map<String, int> countsByClass = {};
          for (final log in logs) {
            final classType = (log['classType'] ?? '').toString();
            if (classType.isEmpty) continue;
            countsByClass[classType] = (countsByClass[classType] ?? 0) + 1;
          }

          final spots = <FlSpot>[];
          for (int i = 0; i < classNames.length; i++) {
            final count = countsByClass[classNames[i]] ?? 0;
            spots.add(FlSpot(i.toDouble(), count.toDouble()));
          }

          return SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  // Graph Section: inline detections-per-class chart (no box)
                  Align(
                    alignment: Alignment.topLeft,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Text(
                          'Detections per class',
                          style: GoogleFonts.poppins(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Shows how many times each class was detected',
                          style: GoogleFonts.poppins(fontSize: 12, color: Colors.black54),
                        ),
                        const SizedBox(height: 16),
                        SizedBox(
                          height: 260,
                          child: classNames.isEmpty
                              ? Center(
                                  child: Text(
                                    'No logs yet to display.',
                                    style: GoogleFonts.poppins(fontSize: 14),
                                  ),
                                )
                              : Padding(
                                  padding: const EdgeInsets.symmetric(horizontal: 8),
                                  child: LineChart(
                                    LineChartData(
                                      minX: 0,
                                      maxX: spots.isNotEmpty ? spots.last.x : 0,
                                      minY: 0,
                                      maxY: spots.isNotEmpty
                                          ? spots
                                                  .map((s) => s.y)
                                                  .reduce((a, b) => a > b ? a : b) +
                                              1
                                          : 1,
                                      gridData: FlGridData(show: true),
                                      borderData: FlBorderData(
                                        show: true,
                                        border: const Border(
                                          left: BorderSide(color: Colors.black54, width: 1),
                                          bottom: BorderSide(color: Colors.black54, width: 1),
                                          right: BorderSide(color: Colors.transparent),
                                          top: BorderSide(color: Colors.transparent),
                                        ),
                                      ),
                                      titlesData: FlTitlesData(
                                        topTitles:
                                            const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                                        rightTitles:
                                            const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                                        leftTitles: AxisTitles(
                                          sideTitles: SideTitles(
                                            showTitles: true,
                                            reservedSize: 32,
                                            getTitlesWidget: (value, meta) {
                                              if (value % 1 != 0) {
                                                return const SizedBox.shrink();
                                              }
                                              return Text(
                                                value.toInt().toString(),
                                                style: GoogleFonts.poppins(fontSize: 10),
                                              );
                                            },
                                          ),
                                        ),
                                        bottomTitles: AxisTitles(
                                          sideTitles: SideTitles(
                                            showTitles: true,
                                            reservedSize: 40,
                                            interval: 1,
                                            getTitlesWidget: (value, meta) {
                                              final index = value.toInt();
                                              if (index < 0 || index >= classNames.length) {
                                                return const SizedBox.shrink();
                                              }
                                              final fullLabel = classNames[index];
                                              var label = fullLabel
                                                  .replaceAll(' Mushroom', '')
                                                  .replaceAll(' Fungus', '');
                                              return Padding(
                                                padding: const EdgeInsets.only(top: 4),
                                                child: Transform.rotate(
                                                  angle: -math.pi / 4,
                                                  child: Text(
                                                    label,
                                                    style: GoogleFonts.poppins(fontSize: 9),
                                                    textAlign: TextAlign.center,
                                                    overflow: TextOverflow.ellipsis,
                                                  ),
                                                ),
                                              );
                                            },
                                          ),
                                        ),
                                      ),
                                      lineTouchData: LineTouchData(
                                        touchTooltipData: LineTouchTooltipData(
                                          getTooltipItems: (touchedSpots) {
                                            return touchedSpots
                                                .map((barSpot) => LineTooltipItem(
                                                      barSpot.y.toInt().toString(),
                                                      TextStyle(
                                                        color: Colors.white,
                                                        fontSize: 12,
                                                        fontWeight: FontWeight.w600,
                                                      ),
                                                    ))
                                                .toList();
                                          },
                                        ),
                                      ),
                                      lineBarsData: [
                                        LineChartBarData(
                                          spots: spots,
                                          isCurved: true,
                                          color: Theme.of(context).primaryColor,
                                          barWidth: 3,
                                          dotData: FlDotData(show: true),
                                          belowBarData: BarAreaData(
                                            show: true,
                                            color: Theme.of(context)
                                                .primaryColor
                                                .withOpacity(0.2),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
              const SizedBox(height: 16),
              
              // History Section
              GestureDetector(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => HistoryLogsPage(logs: logs)),
                  );
                },
                child: Container(
                  width: double.infinity,
                  height: 330,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.black, width: 3),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 4,
                        offset: const Offset(2, 2),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Text(
                              'History',
                              style: GoogleFonts.poppins(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Tap here for more details',
                                style: GoogleFonts.poppins(
                                  fontSize: 12,
                                  color: Colors.black54,
                                ),
                                textAlign: TextAlign.right,
                              ),
                            ),
                            const SizedBox(width: 4),
                            const Icon(
                              Icons.arrow_forward_ios,
                              size: 24,
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        // Preview of first 3 logs
                        Column(
                          children: [
                            // Table Header
                            Container(
                              padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
                              decoration: BoxDecoration(
                                color: Theme.of(context).primaryColor,
                                borderRadius: const BorderRadius.only(
                                  topLeft: Radius.circular(8),
                                  topRight: Radius.circular(8),
                                ),
                              ),
                              child: Row(
                                children: [
                                  Expanded(
                                    flex: 3,
                                    child: Text(
                                      'Class Type',
                                      style: GoogleFonts.poppins(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold,
                                        fontSize: 12,
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                  ),
                                  Expanded(
                                    flex: 2,
                                    child: Text(
                                      'Accuracy',
                                      style: GoogleFonts.poppins(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold,
                                        fontSize: 12,
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                  ),
                                  Expanded(
                                    flex: 3,
                                    child: Text(
                                      'Time',
                                      style: GoogleFonts.poppins(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold,
                                        fontSize: 12,
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            // Show at most first 2 logs as preview (full list is on detail page)
                            ...logs.take(3).toList().asMap().entries.map((entry) {
                              final index = entry.key;
                              final log = entry.value;
                              return Container(
                                padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
                                decoration: BoxDecoration(
                                  color: index % 2 == 0 ? Colors.grey[100] : Colors.white,
                                  border: Border(
                                    bottom: BorderSide(color: Colors.grey[300]!, width: 1),
                                  ),
                                ),
                                child: Row(
                                  children: [
                                    Expanded(
                                      flex: 3,
                                      child: Text(
                                        log['classType'],
                                        style: GoogleFonts.poppins(fontSize: 12),
                                      ),
                                    ),
                                    Expanded(
                                      flex: 2,
                                      child: Text(
                                        '${log['accuracyRate'].toStringAsFixed(1)}%',
                                        style: GoogleFonts.poppins(fontSize: 12),
                                        textAlign: TextAlign.center,
                                      ),
                                    ),
                                    Expanded(
                                      flex: 3,
                                      child: Text(
                                        log['time'],
                                        style: GoogleFonts.poppins(fontSize: 10),
                                        textAlign: TextAlign.right,
                                        maxLines: 2,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            }),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}

// Graph Detail Page
class GraphDetailPage extends StatelessWidget {
  final List<Map<String, dynamic>> logs;

  const GraphDetailPage({super.key, required this.logs});

  @override
  Widget build(BuildContext context) {
    // Known classes (show all of them on the x-axis, even if count is 0)
    const classNames = [
      'Button Mushroom',
      'Oyster Mushroom',
      'Enoki Mushroom',
      'Morel Mushroom',
      'Chanterelle Mushroom',
      'Black Trumpet Mushroom',
      'Fly Agaric Mushroom',
      'Reishi Mushroom',
      'Coral Fungus',
      'Bleeding Tooth Fungus',
    ];

    // Group logs by class type and count how many detections each class has
    final Map<String, int> countsByClass = {};
    for (final log in logs) {
      final classType = (log['classType'] ?? '').toString();
      if (classType.isEmpty) continue;
      countsByClass[classType] = (countsByClass[classType] ?? 0) + 1;
    }
    final spots = <FlSpot>[];
    for (int i = 0; i < classNames.length; i++) {
      final count = countsByClass[classNames[i]] ?? 0;
      spots.add(FlSpot(i.toDouble(), count.toDouble()));
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Graph',
          style: GoogleFonts.titanOne(
            fontSize: 28,
            color: Colors.white,
          ),
        ),
        centerTitle: true,
        backgroundColor: Theme.of(context).primaryColor,
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: classNames.isEmpty
            ? Center(
                child: Text(
                  'No logs yet to display.',
                  style: GoogleFonts.poppins(fontSize: 16),
                ),
              )
            : Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    'Detections per class',
                    style: GoogleFonts.poppins(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Shows how many times each class was detected',
                    style: GoogleFonts.poppins(fontSize: 12, color: Colors.grey[700]),
                  ),
                  const SizedBox(height: 16),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 8),
                    child: SizedBox(
                      height: 260,
                      child: LineChart(
                        LineChartData(
                          minX: 0,
                          maxX: spots.isNotEmpty ? spots.last.x : 0,
                          minY: 0,
                          maxY: spots.isNotEmpty
                              ? spots.map((s) => s.y).reduce((a, b) => a > b ? a : b) + 1
                              : 1,
                          gridData: FlGridData(show: true),
                          borderData: FlBorderData(
                            show: true,
                            border: const Border(
                              left: BorderSide(color: Colors.black54, width: 1),
                              bottom: BorderSide(color: Colors.black54, width: 1),
                              right: BorderSide(color: Colors.transparent),
                              top: BorderSide(color: Colors.transparent),
                            ),
                          ),
                          titlesData: FlTitlesData(
                            topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                            rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                            leftTitles: AxisTitles(
                              sideTitles: SideTitles(
                                showTitles: true,
                                reservedSize: 32,
                                getTitlesWidget: (value, meta) {
                                  // Only show integer steps (0,1,2,...)
                                  if (value % 1 != 0) {
                                    return const SizedBox.shrink();
                                  }
                                  return Text(
                                    value.toInt().toString(),
                                    style: GoogleFonts.poppins(fontSize: 10),
                                  );
                                },
                              ),
                            ),
                            bottomTitles: AxisTitles(
                              sideTitles: SideTitles(
                                showTitles: true,
                                reservedSize: 40,
                                interval: 1,
                                getTitlesWidget: (value, meta) {
                                  final index = value.toInt();
                                  if (index < 0 || index >= classNames.length) {
                                    return const SizedBox.shrink();
                                  }
                                  // Short display label: remove common suffixes like 'Mushroom' and 'Fungus'
                                  final fullLabel = classNames[index];
                                  var label = fullLabel
                                      .replaceAll(' Mushroom', '')
                                      .replaceAll(' Fungus', '');
                                  return Padding(
                                    padding: const EdgeInsets.only(top: 4),
                                    child: Transform.rotate(
                                      angle: -math.pi / 4,
                                      child: Text(
                                        label,
                                        style: GoogleFonts.poppins(fontSize: 9),
                                        textAlign: TextAlign.center,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                  );
                                },
                              ),
                            ),
                          ),
                          lineTouchData: LineTouchData(
                            touchTooltipData: LineTouchTooltipData(
                              getTooltipItems: (touchedSpots) {
                                return touchedSpots
                                    .map((barSpot) => LineTooltipItem(
                                          barSpot.y.toInt().toString(),
                                          TextStyle(
                                            color: Colors.white,
                                            fontSize: 12,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ))
                                    .toList();
                              },
                            ),
                          ),
                          lineBarsData: [
                            LineChartBarData(
                              spots: spots,
                              isCurved: true,
                              color: Theme.of(context).primaryColor,
                              barWidth: 3,
                              dotData: FlDotData(show: true),
                              belowBarData: BarAreaData(
                                show: true,
                                color: Theme.of(context)
                                    .primaryColor
                                    .withOpacity(0.2),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}

// History Logs Detail Page
class HistoryLogsPage extends StatelessWidget {
  final List<Map<String, dynamic>> logs;
  
  const HistoryLogsPage({super.key, required this.logs});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'History',
          style: GoogleFonts.titanOne(
            fontSize: 28,
            color: Colors.white,
          ),
        ),
        centerTitle: true,
        backgroundColor: Theme.of(context).primaryColor,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              // Table Header
              Container(
                padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
                decoration: BoxDecoration(
                  color: Theme.of(context).primaryColor,
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(8),
                    topRight: Radius.circular(8),
                  ),
                ),
                child: Row(
                  children: [
                    Expanded(
                      flex: 3,
                      child: Text(
                        'Class Type',
                        style: GoogleFonts.poppins(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                    Expanded(
                      flex: 2,
                      child: Text(
                        'Accuracy',
                        style: GoogleFonts.poppins(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                    Expanded(
                      flex: 3,
                      child: Text(
                        'Time',
                        style: GoogleFonts.poppins(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
              ),
              // All logs rows
              ...logs.asMap().entries.map((entry) {
                final index = entry.key;
                final log = entry.value;
                return Container(
                  padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
                  decoration: BoxDecoration(
                    color: index % 2 == 0 ? Colors.grey[100] : Colors.white,
                    border: Border(
                      bottom: BorderSide(color: Colors.grey[300]!, width: 1),
                    ),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        flex: 3,
                        child: Text(
                          log['classType'],
                          style: GoogleFonts.poppins(fontSize: 12),
                        ),
                      ),
                      Expanded(
                        flex: 2,
                        child: Text(
                          '${log['accuracyRate'].toStringAsFixed(1)}%',
                          style: GoogleFonts.poppins(fontSize: 12),
                          textAlign: TextAlign.center,
                        ),
                      ),
                      Expanded(
                        flex: 3,
                        child: Text(
                          log['time'],
                          style: GoogleFonts.poppins(fontSize: 10),
                          textAlign: TextAlign.right,
                        ),
                      ),
                    ],
                  ),
                );
              }),
            ],
          ),
        ),
      ),
    );
  }
}
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class TextClassification {
  final String modePath = 'assets/models/text_classification.tflite';
  final String dictionaryPath = 'assets/models/vocab';
  late Map<String, int> _dict;
  final String pad = '<PAD>';
  final String unk = '<UNKNOWN>';
  final String start = '<START';
  late Interpreter interpreter;
  final int sequenceLength = 256;

  TextClassification() {
    loadModel();
    loadDictionary();
  }

  loadModel() async {
    var options = InterpreterOptions();

    if (Platform.isAndroid) {
      options.addDelegate(XNNPackDelegate());
    }

    if (Platform.isIOS) {
      options.addDelegate(GpuDelegate());
    }

    interpreter = await Interpreter.fromAsset(modePath);

    print("Model loaded successfullly");
  }

  loadDictionary() async {
    var dictionary = await rootBundle.loadString(dictionaryPath);
    Map<String, int> dict = {};
    var vocabList = dictionary.split('\n');
    for (var vocab in vocabList) {
      var entery = vocab.trim().split(' ');
      if (entery.length == 2) {
        dict[entery[0]] = int.parse(entery[1]);
      } else {
        entery.insert(0, 'bad_char');
      }
    }
    _dict = dict;
    print("Dictionary loaded successfully");
  }

  List<List<double>> tokenizeText(String text) {
    var chunks = text.split(' ');
    var vecs = List<double>.filled(sequenceLength, _dict[pad]!.toDouble());

    int index = 0;
    if (_dict.containsKey(start)) {
      vecs[index++] = _dict[start]!.toDouble();
    }

    for (var item in chunks) {
      if (index > sequenceLength) {
        break;
      }
      vecs[index++] = _dict.containsKey(item)
          ? _dict[pad]!.toDouble()
          : _dict[unk]!.toDouble();
    }
    return [vecs];
  }

  List<double> classify(String text) {
    List<List<double>> input = tokenizeText(text);
    var output = List<double>.filled(2, 0).reshape([1, 2]);

    interpreter.run(input, output);

    return [output[0][0], output[0][1]];
  }
}

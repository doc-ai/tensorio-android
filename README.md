# tensorio-android
[![Build Status](https://travis-ci.org/doc-ai/TensorIO.svg?branch=master)](https://travis-ci.org/doc-ai/TensorIO)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![](https://jitpack.io/v/doc-ai/tensorio-android.svg)](https://jitpack.io/#doc-ai/tensorio-android)
[![GitHub issues](https://img.shields.io/github/issues/doc-ai/tensorio-android.svg)](https://github.com/doc-ai/tensorio-android/issues)

TensorIO is an Java wrapper for an underlying machine learning library and currently supports TensorFlow Lite. It abstracts the work of copying bytes into and out of tensors and allows you to interract with native types instead, such as numbers, arrays, hashmaps, and pixel buffers.

This implementation is part of the [TensorIO project](https://doc-ai.github.io/tensorio/) with support for machine learning on iOS, Android, and React Native.

With TensorIO you can perform inference in just a few lines of code:

> Java
```java

// Load the image
InputStream bitmap = getAssets().open("picture2.jpg");
Bitmap bMap = BitmapFactory.decodeStream(bitmap);

// load the model
TIOModelBundleManager manager = new TIOModelBundleManager(getApplicationContext(), path);
TIOModelBundle bundle = manager.bundleWithId(bundleId);
TIOModel model = bundle.newModel();
model.load();

// Run the model on the input
float[] result =  (float[]) model.runOn(bMap);

// Show the most likely predictions
String[] labels = ((TIOVectorLayerDescription) model.descriptionOfOutputAtIndex(0)).getLabels();
```


> Kotlin

```kotlin

// Load the image
val bitmap = assets.open("picture2.jpg")
val bMap = BitmapFactory.decodeStream(bitmap)

// load the model
val manager = TIOModelBundleManager(applicationContext, path)
val bundle = manager.bundleWithId(bundleId)
val model = bundle.newModel()
model.load()

// Run the model on the input
val result = model.runOn(bMap) as FloatArray
```

* [ Overview ](#overview)
* [ Example ](#example)
* [ Requirements ](#requirements)
* [ Installation ](#installation)
* [ License ](#license)
* [ Usage ](#usage)
    * [ Add TensorIO to Your Project ](#importing)
    * [ Basic Usage ](#basic-usage)
    * [ Model Bundles ](#model-bundles)
    * [ The Model JSON File ](#model-json)


<a name="overview"></a>
## Overview

TensorIO supports many kinds of models with multiple input and output layers of different shapes and kinds but with minimal boilerplate code. In fact, you can run a variety of models without needing to write any model specific code at all.

Instead, TensorIO relies on a JSON description of the model that you provide. During inference, the library matches incoming data to the model layers that expect it, performing any transformations that are needed and ensuring that the underlying bytes are copied to the right place.  Once inference is complete, the library copies bytes from the output tensors back to native Java types.

The built-in class for working with TensorFlow Lite (TF Lite) models, `TIOTFLiteModel`, includes support for multiple input and output layers; single-valued, vectored, matrix, and image data; pixel normalization and denormalization; and quantization and dequantization of data. In case you require a completely custom interface to a model you may specify your own class in the JSON description, and TensorIO will use it in place of the default class.

Although TensorIO supports both full TensorFlow and TF Lite models, this README will refer to TFLite throughout. Except for small differences in support of data types (`uint8_t`, `float32_t`, etc), the interface is the same.


<a name="example"></a>
## Example

To run the example project, clone the repo, sync the build.gradle in the Example directory and run the Example App.

- See the MainActivity.java for sample code.

<a name="requirements"></a>
## Requirements

TensorIO requires Android 5.1(Lollipop)+ or minSdkVersion 22 or higher 

<a name="installation"></a>
## Installation

TensorIO for Android is available via [github repo](https://github.com/doc-ai/tensorio-android). Add the following to your build.gradle file:

``` build.gradle
implementation 'com.github.doc-ai:tensorio-android:0.1.0'
```

For instructions on how to add dependencies using Jitpack please follow:
https://jitpack.io/#doc-ai/tensorio-android/0.1.0

The .tflite files are compressed by default. Please add the following to the build.gradle file so that the tflite files are not stored compressed in the APK.

```build.gradle
aaptOptions {
    noCompress "tflite"
}
```

<a name="author"></a>
## Author

[doc.ai](https://doc.ai/)


<a name="license"></a>
## License

TensorIO is available under the Apache 2 license. See the LICENSE file for more info.


<a name="usage"></a>
## Usage

<a name="importing"></a>
### Adding TensorIO to Your Project

You can import individual class from the tensorio directory. For importing a bundle:

> Java
```java
import ai.doc.tensorio.TIOModel.TIOModelBundleManager;
import ai.doc.tensorio.TIOModel.TIOModelBundle;


TIOModelBundleManager manager = new TIOModelBundleManager(getApplicationContext(), "");

// load the model
TIOModelBundle bundle = manager.bundleWithId(id);
TIOModel model = bundle.newModel();
model.load();
float[] result =  (float[]) model.runOn(bMap);
```



> Kotlin

```kotlin
import ai.doc.tensorio.TIOModel.TIOModelBundleManager


```
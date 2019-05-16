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
Add the following to the build.gradle file if the build command complains that "More than one file was found with OS independent path".

```build.gradle
packagingOptions {
    pickFirst 'META-INF/ASL-2.0.txt'
    pickFirst 'draftv4/schema'
    pickFirst 'draftv3/schema'
    pickFirst 'META-INF/LICENSE'
    pickFirst 'META-INF/LGPL-3.0.txt'
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

<a name="basic-usage"></a>
### Basic Usage

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

// Run the model on the input
float[] result =  (float[]) model.runOn(bMap);
```



> Kotlin

```kotlin
import ai.doc.tensorio.TIOModel.TIOModelBundleManager


val manager = TIOModelBundleManager(applicationContext, "")

// load the model
val bundle = manager.bundleWithId(id)
val model = bundle.newModel()
model.load()

// Run the model on the input
var result = model.runOn(bMap) as FloatArray
```


<a name="model-bundles"></a>
### Model Bundles

TensorIO currently includes support for TensorFlow Lite (TF Lite) models. Although the library is built with support for other machine learning frameworks in mind, we'll focus on TF Lite models here.

A TF Lite model is contained in a single *.tflite* file. All the operations and weights required to perform inference with a model are included in this file.

However, a model may have other assets that are required to interpret the resulting inference. For example, an ImageNet image classification model will output 1000 values corresponding to the softmax probability that a particular object has been recognized in an image. The model doesn't match probabilities to their labels, for example "rocking chair" or "lakeside", it only outputs numeric values. It is left to us to associate the numeric values with their labels.

Rather than requiring a developer to do this in application space and consequently store the lables in a text file or in some code somewhere in the application, TensorIO wraps models in a bundle and allows model builders to include additional assets in that bundle.

A TensorIO bundle is just a folder with an extension that identifies it as such: *.tiobundle*. Assets may be included in this bundle and then referenced from model specific code. 

*When you use your own models with TensorIO, make sure to put them in a folder with the .tiobundle extension.*

A TensorIO TF Lite bundle has the following directory structure:

```
mymodel.tiobundle
  - model.tflite
  - model.json
  - assets
    - file.txt
    - ...
```

The *model.json* file is required. It describes the interface to your model and includes other metadata about it. More on that below.

The *model.tflite* file is required but may have another name. The bundle must include some *.tflite* file, but its actual name is specified in *model.json*.

The *assets* directory is optional and contains any additional assets required by your specific use case. Those assets may be referenced from *model.json*.

Because image classification is such a common task, TensorIO includes built-in support for it, and no additional code is required. You'll simply need to specify a labels file in the model's JSON description.


<a name="model-json"></a>
### The Model JSON File

One of TensorIO's goals is to reduce the amount of new code required to integrate models into an application.

The primary work of using a model on iOS involves copying bytes of the right length to the right place. TF Lite, for example, is a C++ library, and the input and output tensors are exposed as C style buffers. In order to use a model we must copy byte representations of our input data into these buffers, ask the library to perform inference on those bytes, and then extract the byte representations back out of them.

Model interfaces can vary widely. Some models may have a single input and single output layer, others multiple inputs with a single output, or vice versa. The layers may be of varying shapes, with some layers taking single values, others an array of values, and yet others taking matrices or volumes of higher dimensions. Some models may work on four byte, floating point representations of data, while others use single byte, unsigned integer representations. The latter are called *quantized* models, more on them below.

Consequently, every time we want to try a different model, or even the same model with a slightly different interface, we must modify the code that moves bytes into and out of  buffers.

TensorIO abstracts the work of copying bytes into and out of tensors and replaces that imperative code with a declarative language you already know: JSON.

The *model.json* file in a TensorIO bundle contains metadata about the underlying model as well as a description of the model's input and output layers. TensorIO parses those descriptions and then, when you perform inference with the model, internally handles all the byte copying operations, taking into account layer shapes, data sizes, data transformations, and even output labeling. All you have to do is provide data to the model and ask for the data out of it.

The *model.json* file is the primary point of interaction with the TensorIO library. Any code you write to prepare data for a model and read data from a model will depend on a description of the model's input and output layers that you provide in this file.
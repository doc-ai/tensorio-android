# tensorio-android

<!--
[![Build Status](https://travis-ci.org/doc-ai/TensorIO.svg?branch=master)](https://travis-ci.org/doc-ai/TensorIO)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![](https://jitpack.io/v/doc-ai/tensorio-android.svg)](https://jitpack.io/#doc-ai/tensorio-android)
[![GitHub issues](https://img.shields.io/github/issues/doc-ai/tensorio-android.svg)](https://github.com/doc-ai/tensorio-android/issues)
-->

Tensor/IO for Android is a Java and Kotlin compatible wrapper for machine learning. The library wraps an underlying machine learning framework via the JNI and abstracts the work of copying bytes into and out of tensors, allowing you to interact with native types such as numbers, arrays, hashmaps, and bitmaps instead. Interfaces to your model are described in a declarative manner, and Tensor/IO handles the heavy lifting of transforming and preprocessing the data you give it to run on the underlying model.

With Tensor/IO you can perform inference in just a few lines of code:

**TF Lite Java**

```java
// Load the Model

ModelBundle bundle = ModelBundle.bundleWithAsset(getApplicationContext(), "mobilenet_v2_1.4_224.tiobundle");
Model model = bundle.newModel();

// Load an Image

InputStream stream = testContext.getAssets().open("example-image.jpg");
Bitmap bitmap = BitmapFactory.decodeStream(stream);

// Run the Model

Map<String,Object> output = model.runOn(bitmap);

// Get the Results

Map<String, Float> classification = (Map<String, Float>)output.get("classification");
List<Map.Entry<String, Float>> top5 = ClassificationHelper.topN(classification, 5);
Map.Entry<String, Float> top = top5.get(0);
String label = top.getKey();
```

**TF Lite Kotlin**

```kotlin
// Load the Model

val bundle = ModelBundle.bundleWithAsset(applicationContext, "mobilenet_v2_1.4_224.tiobundle")
val model = bundle.newModel()

// Load an Image

val stream = assets.open("elephant.jpg")
val bitmap = BitmapFactory.decodeStream(stream)

// Run the Model

val output = model.runOn(bitmap)

// Get the Results

val classification = output.get("classification") as MutableMap<String, Float>
val top5 = ClassificationHelper.topN(classification, 5, 0.1f)
val label = top5.get(0).key
```

### TF Lite, TensorFlow, and PyTorch

Tensor/IO currently supports inference with TensorFlow Lite, inference with PyTorch, and inference and training with TensorFlow. This implementation is part of the [Tensor/IO project](https://doc-ai.github.io/tensorio/) with support for machine learning on iOS, Android, and React Native.

For full TensorFlow V2 support, Tensor/IO uses [tensorio-tensorflow-android](https://github.com/doc-ai/tensorio-tensorflow-android), our wrapper library that provides a JNI interface to our [custom build of tensorflow](https://github.com/doc-ai/tensorflow/tree/r2.0.doc.ai-android).

<a name="overview"></a>
## Overview

Tensor/IO supports many kinds of models with multiple input and output layers of different shapes and kinds but with minimal boilerplate code. In fact, you can run a variety of models without needing to write any model specific code at all.

Instead, Tensor/IO relies on a JSON description of the model that you provide. During inference, the library matches incoming data to the model layers that expect it, performing any transformations that are needed and ensuring that the underlying bytes are copied to the right place.  Once inference is complete, the library copies bytes from the output tensors back to native Java types.

The built-in class for working with TensorFlow Lite (TF Lite) models, `TFLiteModel`, includes support for multiple input and output layers; single-valued, vectored, matrix, and image data; pixel normalization and denormalization; and quantization and dequantization of data. In case you require a completely custom interface to a model you may specify your own class in the JSON description, and Tensor/IO will use it in place of the default class.

Although Tensor/IO supports both full TensorFlow and TF Lite models, this README will refer to TFLite throughout. Except for small differences in support of data types (`uint8_t`, `float32_t`, `int32_t`, `int64_t`, etc), the interface is the same, with the addition of training for TensorFlow models (see below).

<a name="example"></a>
## Example

To run the example project, clone the repo, sync the build.gradle in one of the example directories and run the example app. See specifically the MainActivity in either of the TF Lite or TensorFlow examples.

<a name="requirements"></a>
## Requirements

Tensor/IO requires Android 5.1(Lollipop)+ or minSdkVersion 22 or higher.

You should always target Java 8 as a compile option for compatibility, and if you are using an SKD Version less than 26 you must add desugaring instructions to your app's gradle.build file:

```gradle
compileOptions {
  coreLibraryDesugaringEnabled true
  sourceCompatibility JavaVersion.VERSION_1_8
  targetCompatibility JavaVersion.VERSION_1_8
}

dependencies {
  coreLibraryDesugaring 'com.android.tools:desugar_jdk_libs:1.0.10'
  ...
}
```


<a name="installation"></a>
## Installation

Tensor/IO for Android is available via [github repo](https://github.com/doc-ai/tensorio-android) using JitPack. For instructions on how to add dependencies using Jitpack to your build.gradle file follow https://jitpack.io/#doc-ai/tensorio-android/0.12.0

### TF Lite

TF Lite installation looks something like:

```gradle
allprojects {
  repositories {
    maven { url 'https://jitpack.io' }
  }
}

dependencies {
  implementation 'com.github.doc-ai.tensorio-android:core:0.12.0'
  implementation 'com.github.doc-ai.tensorio-android:tflite:0.12.0'
  ...
}
```

TF Lite binaries are compressed by default, which we don't want. Add the following to the build.gradle file so that the tflite files are not compressed in the APK.

```gradle
aaptOptions {
  noCompress "tflite"
}
```

If you encounter the error *"More than one file was found with OS independent path"*, add the following to the build.gradle:

```gradle
packagingOptions {
  pickFirst 'META-INF/ASL-2.0.txt'
  pickFirst 'draftv4/schema'
  pickFirst 'draftv3/schema'
  pickFirst 'META-INF/LICENSE'
  pickFirst 'META-INF/LGPL-3.0.txt'
}
```

### TensorFlow

TensorFlow dependency requirements are similar but the additional options are unnecessary. It will look like:

```gradle
allprojects {
  repositories {
    maven { url 'https://jitpack.io' }
  }
}

dependencies {
  implementation 'com.github.doc-ai.tensorio-android:core:0.12.0'
  implementation 'com.github.doc-ai.tensorio-android:tensorflow:0.12.0'
  ...
}
```

### PyTorch

The PyTorch dependency requirements are like the TensorFlow ones:

```gradle
allprojects {
  repositories {
    maven { url 'https://jitpack.io' }
  }
}

dependencies {
  implementation 'com.github.doc-ai.tensorio-android:core:0.12.0'
  implementation 'com.github.doc-ai.tensorio-android:pytorch:0.12.0'
  ...
}
```

<a name="author"></a>
## Author

[doc.ai](https://doc.ai/)

<a name="license"></a>
## License

Tensor/IO is available under the Apache 2 license. See the LICENSE file for more info.

<a name="usage"></a>
## Usage

<a name="basic-usage"></a>
### TF Lite Basic Usage

Add a tensor/io compatible model to your project's assets directory and run the following lines of code. Here we're using the *mobilenet_v2_1.4_224.tiobundle* model, which you can find in the example project's assets directory.

**Java**

```java
// Load the Model

ModelBundle bundle = new ModelBundle.bundleWithAsset(getApplicationContext(), "mobilenet_v2_1.4_224.tiobundle");
Model model = bundle.newModel();

// Load an Image

InputStream stream = testContext.getAssets().open("example-image.jpg");
Bitmap bitmap = BitmapFactory.decodeStream(stream);

// Run the Model

Map<String,Object> output = model.runOn(bitmap);

// Get the Results

Map<String, Float> classification = (Map<String, Float>)output.get("classification");
List<Map.Entry<String, Float>> top5 = ClassificationHelper.topN(classification, 5);
Map.Entry<String, Float> top = top5.get(0);
String label = top.getKey();
```

**Kotlin**

```kotlin
// Load the Model

val bundle = ModelBundle.bundleWithAsset(applicationContext, "mobilenet_v2_1.4_224.tiobundle")
val model = bundle.newModel()

// Load an Image

val stream = assets.open("elephant.jpg")
val bitmap = BitmapFactory.decodeStream(stream)

// Run the Model

val output = model.runOn(bitmap)

// Get the Results

val classification = output.get("classification") as MutableMap<String, Float>
val top5 = ClassificationHelper.topN(classification, 5, 0.1f)
val label = top5.get(0).key
```

<a name="model-bundles"></a>
### Model Bundles

For additional information about Model Bundles and converting your models to the TF Lite format, refer to the [Tensor/IO Wiki](https://github.com/doc-ai/tensorio-ios/wiki):

- [Packaging Models](https://github.com/doc-ai/tensorio-ios/wiki/Packaging-Models)
- [TF Lite Backend](https://github.com/doc-ai/tensorio-ios/wiki/TensorFlow-Lite-Backend)

A TF Lite model is contained in a single *.tflite* file. All the operations and weights required to perform inference with a model are included in this file.

However, a model may have other assets that are required to interpret the resulting inference. For example, an ImageNet image classification model will output 1000 values corresponding to the softmax probability that a particular object has been recognized in an image. The model doesn't match probabilities to their labels, for example "rocking chair" or "lakeside", it only outputs numeric values. It is left to us to associate the numeric values with their labels.

Rather than requiring a developer to do this in application space and consequently store the lables in a text file or in some code somewhere in the application, Tensor/IO wraps models in a bundle and allows model builders to include additional assets in that bundle.

A Tensor/IO bundle is just a folder with an extension that identifies it as such: *.tiobundle*. Assets may be included in this bundle and then referenced from model specific code. 

*When you use your own models with Tensor/IO, make sure to put them in a folder with the .tiobundle extension.*

A Tensor/IO TF Lite bundle has the following directory structure:

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

Because image classification is such a common task, Tensor/IO includes built-in support for it, and no additional code is required. You'll simply need to specify a labels file in the model's JSON description.


<a name="model-json"></a>
### The Model JSON File

One of Tensor/IO's goals is to reduce the amount of new code required to integrate models into an application.

The primary work of using a model on iOS involves copying bytes of the right length to the right place. TF Lite, for example, is a C++ library, and the input and output tensors are exposed as C style buffers. In order to use a model we must copy byte representations of our input data into these buffers, ask the library to perform inference on those bytes, and then extract the byte representations back out of them.

Model interfaces can vary widely. Some models may have a single input and single output layer, others multiple inputs with a single output, or vice versa. The layers may be of varying shapes, with some layers taking single values, others an array of values, and yet others taking matrices or volumes of higher dimensions. Some models may work on four byte, floating point representations of data, while others use single byte, unsigned integer representations. The latter are called *quantized* models, more on them below.

Consequently, every time we want to try a different model, or even the same model with a slightly different interface, we must modify the code that moves bytes into and out of  buffers.

Tensor/IO abstracts the work of copying bytes into and out of tensors and replaces that imperative code with a declarative language you already know: JSON.

The *model.json* file in a Tensor/IO bundle contains metadata about the underlying model as well as a description of the model's input and output layers. Tensor/IO parses those descriptions and then, when you perform inference with the model, internally handles all the byte copying operations, taking into account layer shapes, data sizes, data transformations, and even output labeling. All you have to do is provide data to the model and ask for the data out of it.

The *model.json* file is the primary point of interaction with the Tensor/IO library. Any code you write to prepare data for a model and read data from a model will depend on a description of the model's input and output layers that you provide in this file.

### TensorFlow Basic Usage

Performing inference with a full TensorFlow model is the same as it is with TF Lite but more datatypes such as the Java primitives `int` and `long` are supported.

The model bundle structure is slightly different. Tensor/IO TensorFlow consumes models exported in the SavedModel format, so that a bundle will look like:

```
mymodel.tiobundle
  - model.json
  - predict
    - saved_model.pb
    - variables
      - variables.data-00000-of-00001
      - variables.index
  - assets
    - file.txt
    - ...
```

For more information about the SavedModel format, see Tensor/IO's documentation on using the [TensorFlow Backend](https://github.com/doc-ai/tensorio-ios/wiki/TensorFlow-Backend).

Note that TensorFlow models must be read from file paths and not from within the packaged application, so that model Assets that are packaged with your app must be copied to a files directory before being accessed.

### Training with TensorFlow

Tensor/IO TensorFlow supports training of models on device with the underlying TensorFlow library. At doc.ai we use this capabaility in support of our federated machine learning efforts.

Training a model is no different than running inference with it. You will prepare your inputs to the model, train on those inputs for some number of epochs, and read the output of the model when finished. Training inputs to your model will typically be both inputs and their labels while the output will typically be the value of some loss function.

**Training with TensorFlow**

```java
// Prepare Model

ModelBundle bundle = bundleForFile("cats-vs-dogs-train.tiobundle");
TrainableModel model = (TrainableModel) bundle.newModel();
model.load();

// Prepare Input

InputStream stream = testContext.getAssets().open("cat.jpg");
Bitmap bitmap = BitmapFactory.decodeStream(stream);

float[] labels = {
        0
};

Map<String, Object> input = new HashMap<String, Object>();
input.put("image", bitmap);
input.put("labels", labels);

// Train Model

float[] losses = new float[4];
int epochs = 4;

for (int epoch = 0; epoch < epochs; epoch++) {

    Map<String,Object> output = model.trainOn(input);
    assertNotNull(output);

    float loss = ((float[]) Objects.requireNonNull(output.get("sigmoid_cross_entropy_loss/value")))[0];
    losses[epoch] = loss;
}
```

In this example the loss value from each epoch is captured in an array and upon inspection you should see the loss value decreasing. Note that we use a utility method `bundleForFile` here which first copies the model from the app's Assets directory to a file directory for use.

**Batched Training**

Training on a single item at a time will usualy not be want you want to do, so a facility for training on batches of data is provided. Your models must support batched training, which means the first dimension of your input tensors will have a value of `-1`. That dimension's its actual value will be set during the call to training based on the size of the batch provided.

```java
// Prepare Model

ModelBundle bundle = bundleForFile("cats-vs-dogs-train.tiobundle");
TrainableModel model = (TrainableModel) bundle.newModel();
model.load();

// Prepare Input

InputStream stream1 = testContext.getAssets().open("cat.jpg");
Bitmap bitmap1 = BitmapFactory.decodeStream(stream1);

float[] labels1 = {
        0
};

Batch.Item input1 = new Batch.Item();
input1.put("image", bitmap1);
input1.put("labels", labels1);

InputStream stream2 = testContext.getAssets().open("dog.jpg");
Bitmap bitmap2 = BitmapFactory.decodeStream(stream2);

float[] labels2 = {
        1
};

Batch.Item input2 = new Batch.Item();
input2.put("image", bitmap2);
input2.put("labels", labels2);

String[] keys = {"image", "labels"};
Batch batch = new Batch(keys);
batch.add(input1);
batch.add(input2);

// Train Model

float[] losses = new float[4];
int epochs = 4;

for (int epoch = 0; epoch < epochs; epoch++) {

    Map<String,Object> output = model.trainOn(batch);
    assertNotNull(output);

    float loss = ((float[]) Objects.requireNonNull(output.get("sigmoid_cross_entropy_loss/value")))[0];
    losses[epoch] = loss;
}
```

Notice that the batch is built up from batch items, which are just maps of key-value pairs. Once again the loss value from each epoch of training is captured and upon inspection you should see the loss decreasing.

### Exporting Model Updates

When you are finished training you will probably want to export the updated model weights for use in some manner. Before calling `model.unload()` simply export the weights to some File path:

```java
File exportDir = exportForFile("cats-vs-dogs");
model.exportTo(exportDir);
```

Provide an existing directory for the export, here we are creating one up in the `exportForFile` call that you can find in the tests, and we pass the directory to the `exportTo` function. That's it. For TensorFlow models you will find a checkpoint created in that directory composed of two files:

```
checkpoint.index
checkpoint.data-00000-of-00001
```

These two files can be treated like the variables of an exported SavedModel in TensorFlow python and used as such. Given the file structure of an exported model:

```
saved_model.pb
variables/
  variables.index
  variables.data-00000-of-00001
```

Simply replace the two variables file with the corresponding checkpoint files produced by the on-device export and load the model as you normally would.
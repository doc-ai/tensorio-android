/*
 * TIOModelJSONParsing.java
 * TensorIO
 *
 * Created by Philip Dow on 7/6/2020
 * Copyright (c) 2020 - Present doc.ai (http://doc.ai)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.doc.tensorio.core.modelbundle;

import ai.doc.tensorio.core.layerinterface.DataType;
import ai.doc.tensorio.core.layerinterface.StringLayerDescription;
import ai.doc.tensorio.core.model.ImageVolume;
import ai.doc.tensorio.core.model.PixelFormat;
import ai.doc.tensorio.core.utilities.AndroidAssets;
import ai.doc.tensorio.core.utilities.FileIO;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ai.doc.tensorio.core.data.Dequantizer;
import ai.doc.tensorio.core.data.Quantizer;
import ai.doc.tensorio.core.data.PixelDenormalizer;
import ai.doc.tensorio.core.data.PixelNormalizer;
import ai.doc.tensorio.core.layerinterface.LayerInterface;
import ai.doc.tensorio.core.layerinterface.PixelBufferLayerDescription;
import ai.doc.tensorio.core.layerinterface.VectorLayerDescription;

import static ai.doc.tensorio.core.layerinterface.LayerInterface.*;

public abstract class JSONParsing {

    /**
     * Key identifying an array (vector) layer
     */

    private static final String TENSOR_TYPE_VECTOR = "array";

    /**
     * Key identifying an image layer
     */

    private static final String TENSOR_TYPE_IMAGE = "image";

    /**
     * Key identifying a string (bytes) layer
     */

    private static final String TENSOR_TYPE_STRING = "string";

    /**
     * Enumerates through the JSON description of a model's inputs or outputs and constructs a
     * `TIOLayerInterface` for each one.
     * @param modelBundle The model bundle whose layer descriptions are being parsed.
     *                    May be `null` if descriptions are being parsed from something other
     *                    than a bundle.
     * @param io A JSONArray of Maps describing the model's input or output layers
     * @param mode `TIOLayerInterfaceMode` one of input, output, or placeholder, describing the
     *             kind of layer this is.
     * @return A List of `TIOLayerInterface` matching the descriptions.
     * @throws JSONException
     * @throws ModelBundleException
     * @throws IOException
     */

    public static List<LayerInterface> parseIO(@Nullable ModelBundle modelBundle, @NonNull JSONArray io, Mode mode) throws JSONException, ModelBundleException, IOException {
        ArrayList<LayerInterface> interfaces = new ArrayList<>();
        boolean isQuantized = modelBundle.isQuantized(); // Always false if modelBundle is nil

        for (int i = 0; i < io.length(); i++) {
            JSONObject jsonObject = io.getJSONObject(i);
            String type = jsonObject.getString("type");
            String name = jsonObject.getString("name");

            LayerInterface layerInterface;

            switch (type) {
                case TENSOR_TYPE_VECTOR:
                    layerInterface = parseTIOVectorDescription(modelBundle, jsonObject, mode, isQuantized);
                    break;
                case TENSOR_TYPE_IMAGE:
                    layerInterface = parseTIOPixelBufferDescription(jsonObject, mode, isQuantized);
                    break;
                case TENSOR_TYPE_STRING:
                    layerInterface = parseTIOStringDescription(jsonObject, mode, isQuantized);
                    break;
                default:
                    throw new ModelBundleException("Unsupported input layer type: " + type);
            }

            interfaces.add(layerInterface);
        }

        return interfaces;
    }

    /**
     * Parses the JSON description of a vector input or output.
     *
     * Handles a vector, matrix, or other multidimensional array (tensor), described as a
     * one dimensional unrolled vector with an optional labels entry.
     *
     * @param modelBundle `The ModelBundel` that is being parsed, needed to derive a path to the labels file.
     *                    May be `nil` if descriptions are being parsed from something other than a bundle.
     * @param dict The JSON description in `JSONObject` format.
     * @param mode One of the TIOLayerInterface.mode values
     * @param quantized `true` if the layer expects or returns quantized bytes, `false` otherwise.
     *
     * @return TIOLayerInterface An interface that describes this vector input or output.
     */

    public static LayerInterface parseTIOVectorDescription(@Nullable ModelBundle modelBundle, @NonNull JSONObject dict, Mode mode, boolean quantized) throws JSONException, ModelBundleException, IOException {
        int[] shape = parseIntArray(dict.getJSONArray("shape"));
        String name = dict.getString("name");

        // Labels

        String[] labels = null;

        if (dict.optString("labels", null) != null) {
            try {
                String contents = null;
                // So barf
                switch (modelBundle.getSource()) {
                    case Asset:
                        contents = AndroidAssets.readTextFile(modelBundle.getContext(), modelBundle.pathToAsset(dict.getString("labels")));
                        break;
                    case File:
                        contents = FileIO.readTextFile((modelBundle.fileToAsset(dict.getString("labels"))));
                        break;
                }
                contents = contents.trim();
                labels = contents.split("\\n");
            }
            catch (IOException e){
                throw new ModelBundleException("There was a problem reading the labels file, no labels were loaded", e);
            }
        }

        // Quantization

        Quantizer quantizer = null;

        switch (mode) {
            case Input:
            case Placeholder:
                if ( dict.has("quantize") ) {
                    quantizer = TIODataQuantizerForDict(dict.getJSONObject("quantize"));
                }
                break;
            case Output:
                break;
        }

        // Dequantization

        Dequantizer dequantizer = null;

        switch (mode) {
            case Output:
                if ( dict.has("dequantize") ) {
                    dequantizer = TIODataDequantizerForDict(dict.getJSONObject("dequantize"));
                }
                break;
            case Input:
            case Placeholder:
                break;
        }

        // Interface

        return new LayerInterface(name, mode, new VectorLayerDescription(
                shape,
                labels,
                quantized,
                quantizer,
                dequantizer)
        );
    }

    /**
     * Parses the JSON description of a pixel buffer input or output.
     *
     * Pixel buffers are handled as their own case instead of as a three-dimensional volume because
     * of byte alignment and pixel format conversion requirements.
     *
     * @param dict The JSON description in `JSONObject` format.
     * @param mode One of the TIOLayerInterface.mode values
     * @param quantized `true` if the layer expects or returns quantized bytes, `false` otherwise.
     *
     * @return TIOLayerInterface An interface that describes this pixel buffer input or output.
     */

    public static LayerInterface parseTIOPixelBufferDescription(@NonNull JSONObject dict, Mode mode, boolean quantized) throws ModelBundleException, JSONException {
        String name = dict.getString("name");

        // Image Volume

        ImageVolume imageVolume;

        try {
            int[] shape = parseIntArray(dict.getJSONArray("shape"));
            imageVolume = TIOImageVolumeForShape(shape);
        } catch (JSONException e) {
            throw new ModelBundleException("Expected input.shape array field in model.json, none found", e);
        }

        // Pixel Format

        PixelFormat pixelFormat;

        try {
            pixelFormat = TIOPixelFormatForString(dict.getString("format"));
        } catch (JSONException e) {
            throw new ModelBundleException("Expected input.format string in model.json, none found", e);
        }

        // Normalization

        PixelNormalizer normalizer = null;

        switch (mode) {
            case Input:
            case Placeholder:
                if ( dict.has("normalize") ) {
                    normalizer = TIOPixelNormalizerForDictionary(dict.getJSONObject("normalize"));
                }
                break;
            case Output:
                break;
        }

        // Denormalization

        PixelDenormalizer denormalizer = null;

        switch (mode) {
            case Output:
                if ( dict.has("denormalize") ) {
                    denormalizer = TIOPixelDenormalizerForDictionary(dict.getJSONObject("denormalize"));
                }
                break;
            case Input:
            case Placeholder:
                break;
        }

        // Description

        return new LayerInterface(name, mode, new PixelBufferLayerDescription(
                pixelFormat,
                imageVolume,
                normalizer,
                denormalizer,
                quantized)
        );
    }

    /**
     * Parses the JSON description of a string (bytes) input or output.
     *
     * @param dict The JSON description in `JSONObject` format.
     * @param mode One of the TIOLayerInterface.mode values
     * @param quantized `true` if the layer expects or returns quantized bytes, `false` otherwise.
     *                  This property is ignored for raw string (bytes) layers.
     *
     * @return TIOLayerInterface An interface that describes this string (bytes) input or output.
     */

    public static LayerInterface parseTIOStringDescription(@NonNull JSONObject dict, Mode mode, boolean quantized) throws JSONException, ModelBundleException, IOException {
        int[] shape = parseIntArray(dict.getJSONArray("shape"));
        String name = dict.getString("name");
        String type = dict.getString("type");

        // Data Type

        DataType dtype = null;

        switch (type) {
            case "uint8":
                dtype = DataType.UInt8;
                break;
            case "float32":
                dtype = DataType.Float32;
                break;
            case "int32":
                dtype = DataType.Int32;
                break;
            case "int64":
                dtype = DataType.Int64;
                break;
            default:
                throw new ModelBundleException("Expected input.dtype to be one of [uint8, float32, int32, int64], found " + type);
        }

        return new LayerInterface(name, mode, new StringLayerDescription(
                shape,
                dtype)
        );
    }

    /**
     * Converts a pixel format string such as `"RGB"` or `"BGR"` to a `TIOPixelFormat` pixel format type.
     */

    public static PixelFormat TIOPixelFormatForString(@NonNull String format) throws ModelBundleException {
        switch (format) {
            case "RGB":
                return PixelFormat.RGB;
            case "BGR":
                return PixelFormat.BGR;
        }
        throw new ModelBundleException("Expected dict.format string to be RGB or BGR in model.json, found " + format);
    }

    /**
     * Parses the `quantization` key of an input description and returns an associated data quantizer.
     */

    public static Quantizer TIODataQuantizerForDict(@Nullable JSONObject dict) throws JSONException, ModelBundleException {
        if (dict == null) {
            return null;
        }

        String standard = dict.optString("standard", null);

        if (standard != null) {
            switch (standard) {
                case "[0,1]":
                    return Quantizer.TIODataQuantizerZeroToOne();
                case "[-1,1]":
                    return Quantizer.TIODataQuantizerNegativeOneToOne();
                default:
                    throw new ModelBundleException("Invalid Quantizer, expected standard quantization to be [0,1] or [1,1]");
            }
        } else {
            if (dict.has("scale") && dict.has("bias")) {
                float scale = (float) dict.getDouble("scale");
                float bias = (float) dict.getDouble("bias");
                return Quantizer.TIODataQuantizerWithQuantization(scale, bias);
            } else {
                throw new ModelBundleException("Invalid Quantizer, expected scale and bias for quantizer");
            }
        }
    }

    /**
     * Parses the `dequantization` key of an output description and returns an associated data dequantizer.
     */

    public static Dequantizer TIODataDequantizerForDict(@Nullable JSONObject dict) throws ModelBundleException, JSONException {
        if (dict == null) {
            return null;
        }

        String standard = dict.optString("standard", null);

        if (standard != null) {
            switch (standard) {
                case "[0,1]":
                    return Dequantizer.TIODataDequantizerZeroToOne();
                case "[-1,1]":
                    return Dequantizer.TIODataDequantizerNegativeOneToOne();
                default:
                    throw new ModelBundleException("Invalid Dequantizer, expected standard dequantization to be [0,1] or [1,1]");
            }
        } else {
            if (dict.has("scale") && dict.has("bias")) {
                float scale = (float) dict.getDouble("scale");
                float bias = (float) dict.getDouble("bias");
                return Dequantizer.TIODataDequantizerWithDequantization(scale, bias);
            } else {
                throw new ModelBundleException("Invalid Dequantizer, expected scale and bias for quantizer");
            }
        }
    }

    /**
     * Parses a string representation of an array of integers into an array of integers
     * @param a The string representation
     * @return The corresponding array of integers
     * @throws JSONException
     */

    public static int[] parseIntArray(@NonNull JSONArray a) throws JSONException {
        int[] result = new int[a.length()];
        for (int i = 0; i < a.length(); i++) {
            result[i] = a.getInt(i);
        }
        return result;
    }

    /**
     * Converts an array of shape values to an `TIOImageVolume`.
     */

    public static ImageVolume TIOImageVolumeForShape(int[] shape) throws ModelBundleException {
        if (shape.length != 3) {
            throw new ModelBundleException("Expected shape with three elements, actual count is " + shape.length);
        }
        if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) {
            throw new ModelBundleException("Invalid image input shape, shape elements can not be <= 0");
        }
        return new ImageVolume(shape[0], shape[1], shape[2]);
    }

    /**
     * Returns the TIOPixelNormalizer given an input dictionary.
     */

    public static PixelNormalizer TIOPixelNormalizerForDictionary(@Nullable JSONObject dict) throws ModelBundleException {
        if (dict == null) {
            return null;
        }

        String normalizerString = dict.optString("standard", null);

        if (normalizerString != null) {
            switch (normalizerString) {
                case "[0,1]":
                    return PixelNormalizer.TIOPixelNormalizerZeroToOne();
                case "[-1,1]":
                    return PixelNormalizer.TIOPixelNormalizerNegativeOneToOne();
                default:
                    throw new ModelBundleException("Expected input.normalizer string to be '[0,1]' or '[-1,1]', actual value is " + normalizerString);
            }
        } else if (dict.has("scale") || dict.has("bias")) {
            float scale = (float)dict.optDouble("scale", 1.0);
            float redBias = (float)dict.optDouble("r", 0.0);
            float greenBias = (float)dict.optDouble("g", 0.0);
            float blueBias = (float)dict.optDouble("b", 0.0);
            return PixelNormalizer.TIOPixelNormalizerPerChannelBias(scale, redBias, greenBias, blueBias);
        } else {
            return null;
        }
    }

    /**
     * Returns the denormalizer for a given input dictionary.
     */

    public static PixelDenormalizer TIOPixelDenormalizerForDictionary(@Nullable JSONObject dict) throws ModelBundleException {
        if (dict == null) {
            return null;
        }

        String normalizerString = dict.optString("standard", null);

        if (normalizerString != null) {
            switch (normalizerString) {
                case "[0,1]":
                    return PixelDenormalizer.TIOPixelDenormalizerZeroToOne();
                case "[-1,1]":
                    return PixelDenormalizer.TIOPixelDenormalizerNegativeOneToOne();
                default:
                    throw new ModelBundleException("Expected input.denormalizer string to be '[0,1]' or '[-1,1]', actual value is " + normalizerString);
            }
        } else if (dict.has("scale") || dict.has("bias")) {
            float scale = (float)dict.optDouble("scale", 1.0);
            float redBias = (float)dict.optDouble("r", 0.0);
            float greenBias = (float)dict.optDouble("g", 0.0);
            float blueBias = (float)dict.optDouble("b", 0.0);
            return PixelDenormalizer.TIOPixelDenormalizerPerChannelBias(scale, redBias, greenBias, blueBias);
        } else {
            return null;
        }
    }

}

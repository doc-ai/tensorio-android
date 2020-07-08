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

package ai.doc.tensorio.TIOModel;

import android.support.annotation.Nullable;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ai.doc.tensorio.TIOData.TIODataDequantizer;
import ai.doc.tensorio.TIOData.TIODataQuantizer;
import ai.doc.tensorio.TIOData.TIOPixelDenormalizer;
import ai.doc.tensorio.TIOData.TIOPixelNormalizer;
import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;
import ai.doc.tensorio.TIOLayerInterface.TIOPixelBufferLayerDescription;
import ai.doc.tensorio.TIOLayerInterface.TIOVectorLayerDescription;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOImageVolume;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOPixelFormat;
import ai.doc.tensorio.TIOUtilities.FileIO;

import static ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface.*;

public abstract class TIOModelJSONParsing {

    /**
     * The name of the directory inside a TensorIO bundle that contains additional data, currently 'assets'.
     */

    private static final String TFMODEL_ASSETS_DIRECTORY = "assets";

    /**
     * Key identifying an array (vector) layer
     */

    private static final String TENSOR_TYPE_VECTOR = "array";

    /**
     * Key identifying an image layer
     */

    private static final String TENSOR_TYPE_IMAGE = "image";

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
     * @throws TIOModelBundleException
     * @throws IOException
     */

    public static List<TIOLayerInterface> parseIO(@Nullable TIOModelBundle modelBundle, JSONArray io, Mode mode) throws JSONException, TIOModelBundleException, IOException {
        ArrayList<TIOLayerInterface> interfaces = new ArrayList<>();
        boolean isQuantized = modelBundle.isQuantized(); // Always false if modelBundle is nil

        for (int i = 0; i < io.length(); i++) {
            JSONObject jsonObject = io.getJSONObject(i);
            String type = jsonObject.getString("type");
            String name = jsonObject.getString("name");

            TIOLayerInterface tioLayerInterface;

            switch (type) {
                case TENSOR_TYPE_VECTOR:
                    tioLayerInterface = parseTIOVectorDescription(modelBundle, jsonObject, mode, isQuantized);
                    break;
                case TENSOR_TYPE_IMAGE:
                    tioLayerInterface = parseTIOPixelBufferDescription(jsonObject, mode, isQuantized);
                    break;
                default:
                    throw new TIOModelBundleException("Unsupported input layer type: " + type);
            }

            interfaces.add(tioLayerInterface);
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

    public static TIOLayerInterface parseTIOVectorDescription(@Nullable TIOModelBundle modelBundle, JSONObject dict, Mode mode, boolean quantized) throws JSONException, TIOModelBundleException, IOException {
        int[] shape = parseIntArray(dict.getJSONArray("shape"));
        String name = dict.getString("name");

        // Labels

        String[] labels = null;

        if (dict.optString("labels", null) != null) {
            try {
                // TODO: Better path building
                String contents = FileIO.readFile(modelBundle.getContext(), modelBundle.getPath() + "/" + TFMODEL_ASSETS_DIRECTORY + "/" + dict.getString("labels"));
                contents = contents.trim();
                labels = contents.split("\\n");
            }
            catch (IOException e){
                throw new TIOModelBundleException("There was a problem reading the labels file, no labels were loaded", e);
            }
        }

        // Quantization

        TIODataQuantizer quantizer = null;

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

        TIODataDequantizer dequantizer = null;

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

        return new TIOLayerInterface(name, mode, new TIOVectorLayerDescription(
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

    public static TIOLayerInterface parseTIOPixelBufferDescription(JSONObject dict, Mode mode, boolean quantized) throws TIOModelBundleException, JSONException {
        String name = dict.getString("name");

        // Image Volume

        TIOImageVolume imageVolume;

        try {
            int[] shape = parseIntArray(dict.getJSONArray("shape"));
            imageVolume = TIOImageVolumeForShape(shape);
        } catch (JSONException e) {
            throw new TIOModelBundleException("Expected input.shape array field in model.json, none found", e);
        }

        // Pixel Format

        TIOPixelFormat pixelFormat;

        try {
            pixelFormat = TIOPixelFormatForString(dict.getString("format"));
        } catch (JSONException e) {
            throw new TIOModelBundleException("Expected input.format string in model.json, none found", e);
        }

        // Normalization

        TIOPixelNormalizer normalizer = null;

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

        TIOPixelDenormalizer denormalizer = null;

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

        TIOLayerInterface layerInterface = new TIOLayerInterface(name, mode, new TIOPixelBufferLayerDescription(
                pixelFormat,
                imageVolume,
                normalizer,
                denormalizer,
                quantized)
        );

        return layerInterface ;
    }

    /**
     * Converts a pixel format string such as `"RGB"` or `"BGR"` to a `TIOPixelFormat` pixel format type.
     */

    public static TIOPixelFormat TIOPixelFormatForString(String format) throws TIOModelBundleException {
        switch (format) {
            case "RGB":
                return TIOPixelFormat.RGB;
            case "BGR":
                return TIOPixelFormat.BGR;
        }
        throw new TIOModelBundleException("Expected dict.format string to be RGB or BGR in model.json, found " + format);
    }

    /**
     * Parses the `quantization` key of an input description and returns an associated data quantizer.
     */

    public static TIODataQuantizer TIODataQuantizerForDict(JSONObject dict) throws JSONException, TIOModelBundleException {
        if (dict == null) {
            return null;
        }

        String standard = dict.optString("standard", null);

        if (standard != null) {
            switch (standard) {
                case "[0,1]":
                    return TIODataQuantizer.TIODataQuantizerZeroToOne();
                case "[-1,1]":
                    return TIODataQuantizer.TIODataQuantizerNegativeOneToOne();
                default:
                    throw new TIOModelBundleException("Invalid Quantizer, expected standard quantization to be [0,1] or [1,1]");
            }
        } else {
            if (dict.has("scale") && dict.has("bias")) {
                float scale = (float) dict.getDouble("scale");
                float bias = (float) dict.getDouble("bias");
                return TIODataQuantizer.TIODataQuantizerWithQuantization(scale, bias);
            } else {
                throw new TIOModelBundleException("Invalid Quantizer, expected scale and bias for quantizer");
            }
        }
    }

    /**
     * Parses the `dequantization` key of an output description and returns an associated data dequantizer.
     */

    public static TIODataDequantizer TIODataDequantizerForDict(JSONObject dict) throws TIOModelBundleException, JSONException {
        if (dict == null) {
            return null;
        }

        String standard = dict.optString("standard", null);

        if (standard != null) {
            switch (standard) {
                case "[0,1]":
                    return TIODataDequantizer.TIODataDequantizerZeroToOne();
                case "[-1,1]":
                    return TIODataDequantizer.TIODataDequantizerNegativeOneToOne();
                default:
                    throw new TIOModelBundleException("Invalid Dequantizer, expected standard dequantization to be [0,1] or [1,1]");
            }
        } else {
            if (dict.has("scale") && dict.has("bias")) {
                float scale = (float) dict.getDouble("scale");
                float bias = (float) dict.getDouble("bias");
                return TIODataDequantizer.TIODataDequantizerWithDequantization(scale, bias);
            } else {
                throw new TIOModelBundleException("Invalid Dequantizer, expected scale and bias for quantizer");
            }
        }
    }

    /**
     * Parses a string representation of an array of integers into an array of integers
     * @param a The string representation
     * @return The corresponding array of integers
     * @throws JSONException
     */

    public static int[] parseIntArray(JSONArray a) throws JSONException {
        int[] result = new int[a.length()];
        for (int i = 0; i < a.length(); i++) {
            result[i] = a.getInt(i);
        }
        return result;
    }

    /**
     * Converts an array of shape values to an `TIOImageVolume`.
     */

    public static TIOImageVolume TIOImageVolumeForShape(int[] shape) throws TIOModelBundleException {
        if (shape.length != 3) {
            throw new TIOModelBundleException("Expected shape with three elements, actual count is " + shape.length);
        }
        if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) {
            throw new TIOModelBundleException("Invalid image input shape, shape elements can not be <= 0");
        }
        return new TIOImageVolume(shape[0], shape[1], shape[2]);
    }

    /**
     * Returns the TIOPixelNormalizer given an input dictionary.
     */

    public static TIOPixelNormalizer TIOPixelNormalizerForDictionary(JSONObject dict) throws TIOModelBundleException {
        if (dict == null) {
            return null;
        }

        String normalizerString = dict.optString("standard", null);

        if (normalizerString != null) {
            switch (normalizerString) {
                case "[0,1]":
                    return TIOPixelNormalizer.TIOPixelNormalizerZeroToOne();
                case "[-1,1]":
                    return TIOPixelNormalizer.TIOPixelNormalizerNegativeOneToOne();
                default:
                    throw new TIOModelBundleException("Expected input.normalizer string to be '[0,1]' or '[-1,1]', actual value is " + normalizerString);
            }
        } else if (dict.has("scale") || dict.has("bias")) {
            float scale = (float)dict.optDouble("scale", 1.0);
            float redBias = (float)dict.optDouble("r", 0.0);
            float greenBias = (float)dict.optDouble("g", 0.0);
            float blueBias = (float)dict.optDouble("b", 0.0);
            return TIOPixelNormalizer.TIOPixelNormalizerPerChannelBias(scale, redBias, greenBias, blueBias);
        } else {
            return null;
        }
    }

    /**
     * Returns the denormalizer for a given input dictionary.
     */

    public static TIOPixelDenormalizer TIOPixelDenormalizerForDictionary(JSONObject dict) throws TIOModelBundleException {
        if (dict == null) {
            return null;
        }

        String normalizerString = dict.optString("standard", null);

        if (normalizerString != null) {
            switch (normalizerString) {
                case "[0,1]":
                    return TIOPixelDenormalizer.TIOPixelDenormalizerZeroToOne();
                case "[-1,1]":
                    return TIOPixelDenormalizer.TIOPixelDenormalizerNegativeOneToOne();
                default:
                    throw new TIOModelBundleException("Expected input.denormalizer string to be '[0,1]' or '[-1,1]', actual value is " + normalizerString);
            }
        } else if (dict.has("scale") || dict.has("bias")) {
            float scale = (float)dict.optDouble("scale", 1.0);
            float redBias = (float)dict.optDouble("r", 0.0);
            float greenBias = (float)dict.optDouble("g", 0.0);
            float blueBias = (float)dict.optDouble("b", 0.0);
            return TIOPixelDenormalizer.TIOPixelDenormalizerPerChannelBias(scale, redBias, greenBias, blueBias);
        } else {
            return null;
        }
    }

}

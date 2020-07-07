/*
 * TIOPixelBufferLayerDescription.java
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

package ai.doc.tensorio.TIOLayerInterface;

import android.graphics.Bitmap;

import java.nio.ByteBuffer;

import ai.doc.tensorio.TIOData.TIOPixelDenormalizer;
import ai.doc.tensorio.TIOData.TIOPixelNormalizer;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOImageVolume;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOPixelFormat;
import ai.doc.tensorio.TIOTensorflowLiteModel.TIOTFLiteDataConverter;
import ai.doc.tensorio.TIOTensorflowLiteModel.TIOTFLitePixelBuffer;

/**
 * The description of a pixel buffer input or output layer.
 */

public class TIOPixelBufferLayerDescription extends TIOLayerDescription {

    /**
     * The buffer is responsible for converting data between the model and userland and for providing
     * a backing buffer to the model.
     */

    TIOTFLiteDataConverter converter;

    /**
     * `true` is the layer is quantized, `false` otherwise
     */

    private boolean quantized;

    /**
     * The pixel format of the image data, must be TIOPixelFormat.RGB or TIOPixelFormat.BGR
     */

    private TIOPixelFormat pixelFormat;

    /**
     * The shape of the pixel data, including width, height, and channels
     */

    private TIOImageVolume shape;

    /**
     * A function that normalizes pixel values from a byte range of `[0,255]` to some other
     * floating point range, may be `nil`.
     */

    private TIOPixelNormalizer normalizer;

    /**
     * A function that denormalizes pixel values from a floating point range back to byte values
     * in the range `[0,255]`, may be nil.
     */

    private TIOPixelDenormalizer denormalizer;

    /**
     * Creates a pixel buffer description from the properties parsed in a model.json file.
     *
     * @param pixelFormat  The expected format of the pixels
     * @param shape        The shape of the input image
     * @param normalizer   A function which normalizes the pixel values for an input layer, may be null.
     * @param denormalizer A function which denormalizes pixel values for an output layer, may be null
     * @param quantized    true if this layer expectes quantized values, false otherwise
     */

    public TIOPixelBufferLayerDescription(TIOPixelFormat pixelFormat, TIOImageVolume shape, TIOPixelNormalizer normalizer, TIOPixelDenormalizer denormalizer, boolean quantized) {
        this.pixelFormat = pixelFormat;
        this.shape = shape;
        this.normalizer = normalizer;
        this.denormalizer = denormalizer;
        this.quantized = quantized;

        // TODO: Hardcoded to TFLite
        this.converter = new TIOTFLitePixelBuffer(this);
    }

    //region Getters and Setters

    @Override
    public boolean isQuantized() {
        return quantized;
    }

    public TIOPixelFormat getPixelFormat() {
        return pixelFormat;
    }

    public TIOImageVolume getShape() {
        return shape;
    }

    public TIOPixelNormalizer getNormalizer() {
        return normalizer;
    }

    public TIOPixelDenormalizer getDenormalizer() {
        return denormalizer;
    }

    //endRegion

    //region Buffer Forwarding

    @Override
    public ByteBuffer toByteBuffer(Object o) {
        return converter.toByteBuffer(o, this);
    }

    @Override
    public Bitmap fromByteBuffer(ByteBuffer byteBuffer) {
        return (Bitmap) converter.fromByteBuffer(byteBuffer, this);
    }

    @Override
    public ByteBuffer getBackingByteBuffer() {
        return converter.getBackingByteBuffer();
    }

    // endRegion
}

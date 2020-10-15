/*
 * PixelBufferLayerDescription.java
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

package ai.doc.tensorio.core.layerinterface;

import ai.doc.tensorio.core.data.PixelDenormalizer;
import ai.doc.tensorio.core.data.PixelNormalizer;
import ai.doc.tensorio.core.model.ImageVolume;
import ai.doc.tensorio.core.model.PixelFormat;

/**
 * The description of a pixel buffer input or output layer.
 */

public class PixelBufferLayerDescription extends LayerDescription {

    /**
     * `true` is the layer is quantized, `false` otherwise
     */

    private boolean quantized;

    /**
     * The pixel format of the image data, must be PixelFormat.RGB or PixelFormat.BGR
     */

    private PixelFormat pixelFormat;

    /**
     * The shape of the pixel data, including width, height, and channels
     */

    private ImageVolume shape;

    /**
     * A function that normalizes pixel values from a byte range of `[0,255]` to some other
     * floating point range, may be `nil`.
     */

    private PixelNormalizer normalizer;

    /**
     * A function that denormalizes pixel values from a floating point range back to byte values
     * in the range `[0,255]`, may be nil.
     */

    private PixelDenormalizer denormalizer;

    /**
     * Creates a pixel buffer description from the properties parsed in a model.json file.
     *
     * @param pixelFormat  The expected format of the pixels
     * @param shape        The shape of the input image
     * @param batched      True if the layer supports batched execution, false otherwise
     * @param normalizer   A function which normalizes the pixel values for an input layer, may be null.
     * @param denormalizer A function which denormalizes pixel values for an output layer, may be null
     * @param quantized    true if this layer expectes quantized values, false otherwise
     */

    public PixelBufferLayerDescription(PixelFormat pixelFormat, ImageVolume shape, boolean batched, PixelNormalizer normalizer, PixelDenormalizer denormalizer, boolean quantized) {
        this.pixelFormat = pixelFormat;
        this.shape = shape;
        this.batched = batched;
        this.normalizer = normalizer;
        this.denormalizer = denormalizer;
        this.quantized = quantized;
    }

    //region Getters and Setters

    @Override
    public boolean isQuantized() {
        return quantized;
    }

    public PixelFormat getPixelFormat() {
        return pixelFormat;
    }

    public ImageVolume getShape() {
        return shape;
    }

    public PixelNormalizer getNormalizer() {
        return normalizer;
    }

    public PixelDenormalizer getDenormalizer() {
        return denormalizer;
    }

    @Override
    public int[] getTensorShape() {
        if (isBatched()) {
            return new int[]{-1, shape.height, shape.width, shape.channels};
        } else {
            return new int[]{shape.height, shape.width, shape.channels};
        }
    }

    //endRegion
}

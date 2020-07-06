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
import java.nio.ByteOrder;

import ai.doc.tensorio.TIOData.TIOBuffer;
import ai.doc.tensorio.TIOData.TIOPixelDenormalizer;
import ai.doc.tensorio.TIOData.TIOPixelNormalizer;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOImageVolume;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOPixelFormat;
import ai.doc.tensorio.TIOTensorflowLiteModel.TIOTFLitePixelBuffer;
import ai.doc.tensorio.TIOTensorflowLiteModel.TIOTFLiteVectorBuffer;

/**
 * The description of a pixel buffer input or output layer.
 */

public class TIOPixelBufferLayerDescription extends TIOLayerDescription {

    /**
     * The buffer is responsible for converting data between the model and userland and for providing
     * a backing buffer to the model.
     */

    TIOBuffer buffer;

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
        this.buffer = new TIOTFLitePixelBuffer(this);
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

//    // TODO: incorrectly named method, it may not convert it to float at all! (#29)
//
//    private void intPixelToFloat(int pixelValue, ByteBuffer imgData) {
//        if (quantized) {
//            imgData.put((byte) ((pixelValue >> 16) & 0xFF));
//            imgData.put((byte) ((pixelValue >> 8) & 0xFF));
//            imgData.put((byte) (pixelValue & 0xFF));
//        } else {
//            if (this.normalizer != null) {
//                imgData.putFloat(this.normalizer.normalize((pixelValue >> 16) & 0xFF, 0));
//                imgData.putFloat(this.normalizer.normalize((pixelValue >> 8) & 0xFF, 1));
//                imgData.putFloat(this.normalizer.normalize(pixelValue & 0xFF, 2));
//            } else {
//                imgData.putFloat((pixelValue >> 16) & 0xFF);
//                imgData.putFloat((pixelValue >> 8) & 0xFF);
//                imgData.putFloat(pixelValue & 0xFF);
//            }
//        }
//    }
//
//    private int floatPixelToInt(float r, float g, float b){
//        int rr, gg, bb;
//        if (this.denormalizer != null){
//            rr = this.denormalizer.denormalize(r, 0);
//            gg = this.denormalizer.denormalize(g, 1);
//            bb = this.denormalizer.denormalize(b, 2);
//        }
//        else{
//            rr = (int)r;
//            gg = (int)g;
//            bb = (int)b;
//        }
//        return 0xFF000000 | (rr << 16) & 0x00FF0000| (gg << 8) & 0x0000FF00 | bb & 0x000000FF;
//    }

    //region Buffer Forwarding

    @Override
    public ByteBuffer toByteBuffer(Object o) {
        return buffer.toByteBuffer(o);

//        if (o == null) {
//            throw new NullPointerException("Input to a model can not be null");
//        } else if (!(o instanceof Bitmap)) {
//            throw new IllegalArgumentException("Image input should be bitmap");
//        }
//
//        Bitmap bitmap = (Bitmap) o;
//        if (bitmap.getWidth() != this.shape.width || bitmap.getHeight() != this.shape.height){
//            throw new IllegalArgumentException("Image input has the wrong shape, expected width="+this.shape.width+" and height="+this.shape.height+" got width="+bitmap.getWidth()+" and height="+bitmap.getHeight());
//        }
//
//        int[] intValues = new int[this.shape.width * this.shape.height]; // 4 bytes per int
//
//        buffer.rewind();
//        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight()); // Returns ARGB pixels
//
//        // Convert the image to floating point
//
//        // TODO: Traversing by y-axis on the inside, shouldn't it be by x-axis? doesn't look like it matters
//
//        int pixel = 0;
//        for (int i = 0; i < bitmap.getWidth(); ++i) {
//            for (int j = 0; j < bitmap.getHeight(); ++j) {
//                final int val = intValues[pixel++];
//                intPixelToFloat(val, buffer);
//            }
//        }
//
//        intValues = null;
//
//        return buffer;
    }

    @Override
    public Bitmap fromByteBuffer(ByteBuffer byteBuffer) {
        return (Bitmap) buffer.fromByteBuffer(byteBuffer);

//        int[] intValues = new int[this.shape.width * this.shape.height]; // 4 bytes per int
//
//        byteBuffer.rewind();
//        Bitmap bmp = Bitmap.createBitmap(this.shape.width, this.shape.height, Bitmap.Config.ARGB_8888);
//
//        for (int i=0; i<this.shape.width*this.shape.height; i++) {
//            if (this.quantized) {
//                int r = byteBuffer.get();
//                int g = byteBuffer.get();
//                int b = byteBuffer.get();
//                intValues[i] = 0xFF000000 | (r << 16) & 0x00FF0000| (g << 8) & 0x0000FF00 | b & 0x000000FF;
//            }
//            else {
//                intValues[i] = floatPixelToInt(byteBuffer.getFloat(), byteBuffer.getFloat(), byteBuffer.getFloat());
//            }
//        }
//
//        bmp.setPixels(intValues,0, this.shape.width, 0, 0, this.shape.width, this.shape.height);
//
//        intValues = null;
//
//        return bmp;
    }

    @Override
    public ByteBuffer getBackingByteBuffer() {
        return buffer.getBackingByteBuffer();
    }

    // endRegion
}

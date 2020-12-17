/*
 * BitmapConverter.java
 * TensorIO
 *
 * Created by Sam Leroux on 12/15/2020
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


package ai.doc.tensorio.pytorch.data;

import android.graphics.Bitmap;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.pytorch.Tensor;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import ai.doc.tensorio.core.data.PixelDenormalizer;
import ai.doc.tensorio.core.data.PixelNormalizer;
import ai.doc.tensorio.core.layerinterface.LayerDescription;
import ai.doc.tensorio.core.layerinterface.PixelBufferLayerDescription;
import ai.doc.tensorio.core.model.ImageVolume;

/**
 * The Pytorch pixel data converter transforms bitmaps into Tensors for use as inputs to
 * Pytorch models and Tensor outputs back into bitmaps.
 *
 * The `toByteBuffer` method will scale the Bitmap you provide if necessary using
 * `createScaledBitmap`, or you may scale the Bitmap before hand.
 */

public class BitmapConverter implements ai.doc.tensorio.core.data.Converter, Converter {

    @Override
    public ByteBuffer createBackingBuffer(@NonNull LayerDescription description) {
        ByteBuffer buffer;

        boolean quantized = ((PixelBufferLayerDescription)description).isQuantized();
        ImageVolume shape = ((PixelBufferLayerDescription)description).getShape();

        if (quantized) {
            // Layer expects bytes
            buffer = ByteBuffer.allocateDirect(shape.width * shape.height * shape.channels);
        } else {
            // Layer expects floats
            buffer = ByteBuffer.allocateDirect(shape.width * shape.height * shape.channels * 4);
        }

        buffer.order(ByteOrder.nativeOrder());

        return buffer;
    }

    //@Override
    //public ByteBuffer toByteBuffer(@NonNull Object o, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {

    //}

    @Override
    public Tensor toTensor(@NonNull Object o, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        if (!(o instanceof Bitmap)) {
            throw BadInputException();
        } else {
            return toTensor((Bitmap)o, description, cache);
        }
    }

    /**
     * Converts a Bitmap to a byte buffer. Resizes the Bitmap if necessary using `createScaledBitmap`.
     *
     * @param bitmap The bitmap to convert
     * @param description A description of the layer with instructions on how to make the conversion
     * @param cache A pre-existing byte buffer to use, which will be returned if not null. If a cache
     *              is provided it will be rewound before being used.
     * @return A ByteBuffer ready for use with a TFLite model
     */

    public Tensor toTensor(@NonNull Bitmap bitmap, @NonNull LayerDescription description, @Nullable ByteBuffer cache) {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description);
        buffer.rewind();

        // Acquire needed properties from layer description

        PixelBufferLayerDescription pixelBufferLayerDescription = (PixelBufferLayerDescription) description;
        ImageVolume shape = pixelBufferLayerDescription.getShape();
        boolean quantized = pixelBufferLayerDescription.isQuantized();
        PixelNormalizer normalizer = pixelBufferLayerDescription.getNormalizer();

        // Resize Bitmap

        if (bitmap.getWidth() != shape.width || bitmap.getHeight() != shape.height){
            bitmap = Bitmap.createScaledBitmap(bitmap,shape.width,shape.height,true);
        }

        // Read Bitmap into int array

        int[] intValues = new int[shape.width * shape.height]; // 4 bytes per int

        buffer.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight()); // Returns ARGB pixels

        // Write Individual Pixels to Buffer
        final int pixelsCount = shape.width * shape.height;
        for (int i = 0; i < pixelsCount; i++) {
            final int val = intValues[i];
            writePixelToBuffer(val, i, pixelsCount, buffer, quantized, normalizer);
        }

        intValues = null;

        Tensor tensor = Tensor.fromBlob(buffer.asFloatBuffer(), new long[]{1, shape.channels,shape.height,shape.width});

        return tensor;
    }

    //@Override
    public Bitmap fromByteBuffer(@NonNull Buffer buffer, @NonNull LayerDescription description) {

        // Acquire needed properties from layer description

        PixelBufferLayerDescription pixelBufferLayerDescription = (PixelBufferLayerDescription) description;
        ImageVolume shape = pixelBufferLayerDescription.getShape();
        boolean quantized = pixelBufferLayerDescription.isQuantized();
        PixelDenormalizer denormalizer = pixelBufferLayerDescription.getDenormalizer();

        // Write the buffer into a bitmap

        int[] intValues = new int[shape.width * shape.height]; // 4 bytes per int
        Bitmap bmp = Bitmap.createBitmap(shape.width, shape.height, Bitmap.Config.ARGB_8888);

        buffer.rewind();
        final int pixelsCount = shape.width * shape.height;
        for (int i = 0; i < shape.width * shape.height; i++) {
            intValues[i] = readPixelFromBuffer(i,pixelsCount, buffer, quantized, denormalizer);
        }

        bmp.setPixels(intValues,0, shape.width, 0, 0, shape.width, shape.height);

        intValues = null;

        return bmp;
    }

    //region Utilities

    /**
     * Writes a pixel to a buffer, normalizing and converting it as needed.
     *
     * Before calling this method the first time in a loop, rewind the buffer. The buffer then
     * increments its index with every call to put.
     *
     * @param pixelValue 4 byte pixel value to write with ARGB or BGRA format wit
     * @param buffer The buffer to write to
     * @param quantized true if the buffer expects quantized (byte) data, false otherwise (float)
     * @param normalizer The normalizer than converts a single byte pixel-channel value to a
     *                   floating point value
     */

    public void writePixelToBuffer(int pixelValue, int pixelIndex, int pixelsCount, @NonNull ByteBuffer buffer, boolean quantized, @Nullable PixelNormalizer normalizer) {
        if (quantized) {
            buffer.put(pixelIndex, (byte) ((pixelValue >> 16) & 0xFF));
            buffer.put(pixelIndex+pixelsCount, (byte) ((pixelValue >> 8) & 0xFF));
            buffer.put(pixelIndex+2*pixelsCount, (byte) (pixelValue & 0xFF));
        } else {
            int r = ((pixelValue >> 16) & 0xff);
            int g = ((pixelValue >> 8) & 0xff);
            int b = ((pixelValue) & 0xff);

            if (normalizer != null) {
                float rF = normalizer.normalize(r, 0);
                float gF = normalizer.normalize(g, 1);
                float bF = normalizer.normalize(b, 2);

                buffer.putFloat(pixelIndex*4, rF);
                buffer.putFloat(pixelIndex*4+pixelsCount*4, gF);
                buffer.putFloat(pixelIndex*4+2*pixelsCount*4, bF);
            } else {
                buffer.putFloat(pixelIndex*4, r);
                buffer.putFloat(pixelIndex*4+pixelsCount*4,g);
                buffer.putFloat(pixelIndex*4+2*pixelsCount*4, b);
            }
        }
    }

    /**
     * Reads three bytes of a pixel from a buffer, assuming no alpha channel, and denormalizing and
     * converting the values as needed.
     *
     * Before calling this method the first time in a loop, rewind the buffer. The buffer then
     * increments its index with every call to get.
     *
     * @param buffer The buffer to read from
     * @param quantized True if the buffer contains quantized (byte) data, false otherwise (float)
     * @param denormalizer The denormalizer than converts a floating point pixel-channel value to a
     *                     byte value
     * @return The 4 byte representation of the pixel.
     */

    public int readPixelFromBuffer(int pixelIndex, int pixelsCount, @NonNull Buffer buffer, boolean quantized, @Nullable PixelDenormalizer denormalizer) {
        if (quantized) {
            ByteBuffer byteBuffer = (ByteBuffer)buffer;
            int r = byteBuffer.get();
            int g = byteBuffer.get();
            int b = byteBuffer.get();
            return ( 0xFF000000 | (r << 16) & 0x00FF0000| (g << 8) & 0x0000FF00 | b & 0x000000FF );
        }
        else {
            FloatBuffer floatBuffer = (FloatBuffer)buffer;
            float r = floatBuffer.get(pixelIndex);
            float g = floatBuffer.get(pixelIndex+pixelsCount);
            float b = floatBuffer.get(pixelIndex+2*pixelsCount);

            int rr, gg, bb;

            if (denormalizer != null) {
                rr = denormalizer.denormalize(r, 0);
                gg = denormalizer.denormalize(g, 1);
                bb = denormalizer.denormalize(b, 2);
            } else {
                rr = (int)r;
                gg = (int)g;
                bb = (int)b;
            }

            return 0xFF000000 | (rr << 16) & 0x00FF0000| (gg << 8) & 0x0000FF00 | bb & 0x000000FF;
        }
    }

    // endRegion

    //region Exceptions

    private static IllegalArgumentException BadInputException() {
        return new IllegalArgumentException("Expected Bitmap input to the converter");
    }

    @Override
    public Object fromTensor(@NonNull Tensor t, @NonNull LayerDescription description) {
        boolean quantized = ((PixelBufferLayerDescription)description).isQuantized();
        Buffer buffer;

        if (quantized) {
            // Layer returns bytes
            buffer = ByteBuffer.wrap(t.getDataAsByteArray());
        } else {
            // Layer returns floats
            buffer = FloatBuffer.wrap(t.getDataAsFloatArray());

        }

        return fromByteBuffer(buffer, description);
    }

    //endRegion
}

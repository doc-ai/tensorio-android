/*
 * TIOTFLitePixelDataConverter.java
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

package ai.doc.tensorio.TIOTFLiteData;

import android.graphics.Bitmap;
import android.support.annotation.Nullable;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import ai.doc.tensorio.TIOData.TIODataConverter;
import ai.doc.tensorio.TIOData.TIOPixelDenormalizer;
import ai.doc.tensorio.TIOData.TIOPixelNormalizer;
import ai.doc.tensorio.TIOLayerInterface.TIOLayerDescription;
import ai.doc.tensorio.TIOLayerInterface.TIOPixelBufferLayerDescription;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOImageVolume;

public class TIOTFLitePixelDataConverter implements TIODataConverter, TIOTFLiteDataConverter {

    @Override
    public ByteBuffer createBackingBuffer(TIOLayerDescription description) {
        ByteBuffer buffer;

        boolean quantized = ((TIOPixelBufferLayerDescription)description).isQuantized();
        TIOImageVolume shape = ((TIOPixelBufferLayerDescription)description).getShape();

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

    @Override
    public ByteBuffer toByteBuffer(Object o, TIOLayerDescription description, @Nullable ByteBuffer cache) {

        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer((TIOPixelBufferLayerDescription)description);
        buffer.rewind();

        // Acquire needed properties from layer description

        TIOPixelBufferLayerDescription pixelBufferLayerDescription = (TIOPixelBufferLayerDescription) description;
        TIOImageVolume shape = pixelBufferLayerDescription.getShape();
        boolean quantized = pixelBufferLayerDescription.isQuantized();
        TIOPixelNormalizer normalizer = pixelBufferLayerDescription.getNormalizer();

        if (o == null) {
            throw new NullPointerException("Input to a model can not be null");
        } else if (!(o instanceof Bitmap)) {
            throw new IllegalArgumentException("Image input should be bitmap");
        }

        // Bitmap Operations

        Bitmap bitmap = (Bitmap) o;
        if (bitmap.getWidth() != shape.width || bitmap.getHeight() != shape.height){
            throw new IllegalArgumentException("Image input has the wrong shape, expected width="+shape.width+" " +
                    "and height="+shape.height+" " +
                    "got width="+bitmap.getWidth()+" " +
                    "and height="+bitmap.getHeight());
        }

        int[] intValues = new int[shape.width * shape.height]; // 4 bytes per int

        buffer.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight()); // Returns ARGB pixels

        // Write Individual Pixels to Buffer

        int pixel = 0;
        for (int y = 0; y < bitmap.getHeight(); y++) {
            for (int x = 0; x < bitmap.getWidth(); x++) {
                final int val = intValues[pixel++];
                writePixelToBuffer(val, buffer, quantized, normalizer);
            }
        }

        intValues = null;

        return buffer;
    }

    @Override
    public Bitmap fromByteBuffer(ByteBuffer buffer, TIOLayerDescription description) {

        // Acquire needed properties from layer description

        TIOPixelBufferLayerDescription pixelBufferLayerDescription = (TIOPixelBufferLayerDescription) description;
        TIOImageVolume shape = pixelBufferLayerDescription.getShape();
        boolean quantized = pixelBufferLayerDescription.isQuantized();
        TIOPixelDenormalizer denormalizer = pixelBufferLayerDescription.getDenormalizer();

        // Write the buffer into a bitmap

        int[] intValues = new int[shape.width * shape.height]; // 4 bytes per int
        Bitmap bmp = Bitmap.createBitmap(shape.width, shape.height, Bitmap.Config.ARGB_8888);

        buffer.rewind();

        for (int i = 0; i < shape.width * shape.height; i++) {
            intValues[i] = readPixelFromBuffer(buffer, quantized, denormalizer);
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
     * @param pixelValue 4 byte pixel value to write with ARGB or BGRA format wit
     * @param buffer The buffer to write to
     * @param quantized true if the buffer expects quantized (byte) data, false otherwise (float)
     * @param normalizer The normalizer than converts a single byte pixel-channel value to a
     *                   floating point value
     */

    private void writePixelToBuffer(int pixelValue, ByteBuffer buffer, boolean quantized, @Nullable TIOPixelNormalizer normalizer) {
        if (quantized) {
            buffer.put((byte) ((pixelValue >> 16) & 0xFF));
            buffer.put((byte) ((pixelValue >> 8) & 0xFF));
            buffer.put((byte) (pixelValue & 0xFF));
        } else {
            if (normalizer != null) {
                buffer.putFloat(normalizer.normalize((pixelValue >> 16) & 0xFF, 0));
                buffer.putFloat(normalizer.normalize((pixelValue >> 8) & 0xFF, 1));
                buffer.putFloat(normalizer.normalize(pixelValue & 0xFF, 2));
            } else {
                buffer.putFloat((pixelValue >> 16) & 0xFF);
                buffer.putFloat((pixelValue >> 8) & 0xFF);
                buffer.putFloat(pixelValue & 0xFF);
            }
        }
    }

    /**
     * Reads three bytes of a pixel from a buffer, assuming no alpha channel, and denormalizing and
     * converting the values as needed.
     *
     * Before calling this method the first time in a loop, rewind the buffer. The buffer then
     * increments its index with every call to get.
     * @param buffer The buffer to read from
     * @param quantized True if the buffer contains quantized (byte) data, false otherwise (float)
     * @param denormalizer The denormalizer than converts a floating point pixel-channel value to a
     *                     byte value
     * @return The 4 byte representation of the pixel.
     */

    private int readPixelFromBuffer(ByteBuffer buffer, boolean quantized, @Nullable TIOPixelDenormalizer denormalizer) {
        if (quantized) {
            int r = buffer.get();
            int g = buffer.get();
            int b = buffer.get();
            return ( 0xFF000000 | (r << 16) & 0x00FF0000| (g << 8) & 0x0000FF00 | b & 0x000000FF );
        }
        else {
            float r = buffer.getFloat();
            float g = buffer.getFloat();
            float b = buffer.getFloat();

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
}

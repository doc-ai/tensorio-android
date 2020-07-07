/*
 * TIOTFLitePixelBuffer.java
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

package ai.doc.tensorio.TIOTensorflowLiteModel;

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

public class TIOTFLitePixelBuffer implements TIODataConverter, TIOTFLiteDataConverter {

    /**
     * Backing buffer
     */

    private ByteBuffer buffer;

    public TIOTFLitePixelBuffer(TIOPixelBufferLayerDescription description) {
        boolean quantized = description.isQuantized();
        TIOImageVolume shape = description.getShape();

        if (quantized) {
            // Layer expects bytes
            this.buffer = ByteBuffer.allocateDirect(shape.width * shape.height * shape.channels);
        } else {
            // Layer expects floats
            this.buffer = ByteBuffer.allocateDirect(shape.width * shape.height * shape.channels * 4);
        }

        this.buffer.order(ByteOrder.nativeOrder());
    }

    //region Utilities

    // TODO: incorrectly named method, it may not convert it to float at all! (#29)

    private void intPixelToFloat(int pixelValue, ByteBuffer imgData, boolean quantized, @Nullable TIOPixelNormalizer normalizer) {
        if (quantized) {
            imgData.put((byte) ((pixelValue >> 16) & 0xFF));
            imgData.put((byte) ((pixelValue >> 8) & 0xFF));
            imgData.put((byte) (pixelValue & 0xFF));
        } else {
            if (normalizer != null) {
                imgData.putFloat(normalizer.normalize((pixelValue >> 16) & 0xFF, 0));
                imgData.putFloat(normalizer.normalize((pixelValue >> 8) & 0xFF, 1));
                imgData.putFloat(normalizer.normalize(pixelValue & 0xFF, 2));
            } else {
                imgData.putFloat((pixelValue >> 16) & 0xFF);
                imgData.putFloat((pixelValue >> 8) & 0xFF);
                imgData.putFloat(pixelValue & 0xFF);
            }
        }
    }

    private int floatPixelToInt(float r, float g, float b, @Nullable TIOPixelDenormalizer denormalizer){
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

    // endRegion

    @Override
    public ByteBuffer toByteBuffer(Object o, TIOLayerDescription description) {
        TIOPixelBufferLayerDescription pixelBufferLayerDescription = (TIOPixelBufferLayerDescription) description;
        TIOImageVolume shape = pixelBufferLayerDescription.getShape();
        boolean quantized = pixelBufferLayerDescription.isQuantized();
        TIOPixelNormalizer normalizer = pixelBufferLayerDescription.getNormalizer();

        if (o == null) {
            throw new NullPointerException("Input to a model can not be null");
        } else if (!(o instanceof Bitmap)) {
            throw new IllegalArgumentException("Image input should be bitmap");
        }

        Bitmap bitmap = (Bitmap) o;
        if (bitmap.getWidth() != shape.width || bitmap.getHeight() != shape.height){
            throw new IllegalArgumentException("Image input has the wrong shape, expected width="+shape.width+" and height="+shape.height+" got width="+bitmap.getWidth()+" and height="+bitmap.getHeight());
        }

        int[] intValues = new int[shape.width * shape.height]; // 4 bytes per int

        buffer.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight()); // Returns ARGB pixels

        // Convert the image to floating point

        // TODO: Traversing by y-axis on the inside, shouldn't it be by x-axis? doesn't look like it matters

        int pixel = 0;
        for (int i = 0; i < bitmap.getWidth(); ++i) {
            for (int j = 0; j < bitmap.getHeight(); ++j) {
                final int val = intValues[pixel++];
                intPixelToFloat(val, buffer, quantized, normalizer);
            }
        }

        intValues = null;

        return buffer;
    }

    @Override
    public Bitmap fromByteBuffer(ByteBuffer byteBuffer, TIOLayerDescription description) {
        TIOPixelBufferLayerDescription pixelBufferLayerDescription = (TIOPixelBufferLayerDescription) description;
        TIOImageVolume shape = pixelBufferLayerDescription.getShape();
        boolean quantized = pixelBufferLayerDescription.isQuantized();
        TIOPixelDenormalizer denormalizer = pixelBufferLayerDescription.getDenormalizer();

        int[] intValues = new int[shape.width * shape.height]; // 4 bytes per int

        byteBuffer.rewind();
        Bitmap bmp = Bitmap.createBitmap(shape.width, shape.height, Bitmap.Config.ARGB_8888);

        for (int i = 0; i < shape.width * shape.height; i++) {
            if (quantized) {
                int r = byteBuffer.get();
                int g = byteBuffer.get();
                int b = byteBuffer.get();
                intValues[i] = 0xFF000000 | (r << 16) & 0x00FF0000| (g << 8) & 0x0000FF00 | b & 0x000000FF;
            }
            else {
                intValues[i] = floatPixelToInt(byteBuffer.getFloat(), byteBuffer.getFloat(), byteBuffer.getFloat(), denormalizer);
            }
        }

        bmp.setPixels(intValues,0, shape.width, 0, 0, shape.width, shape.height);

        intValues = null;

        return bmp;
    }

    @Override
    public ByteBuffer getBackingByteBuffer() {
        return buffer;
    }
}

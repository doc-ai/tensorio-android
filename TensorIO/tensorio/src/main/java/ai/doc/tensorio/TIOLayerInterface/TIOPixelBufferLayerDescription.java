package ai.doc.tensorio.TIOLayerInterface;

import android.graphics.Bitmap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import ai.doc.tensorio.TIOData.TIOPixelDenormalizer;
import ai.doc.tensorio.TIOData.TIOPixelNormalizer;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOImageVolume;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOPixelFormat;

/**
 * The description of a pixel buffer input or output layer.
 */

public class TIOPixelBufferLayerDescription extends TIOLayerDescription {

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

    private int[] intValues;
    private ByteBuffer buffer;

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

        if (isQuantized()) {
            this.buffer = ByteBuffer.allocateDirect(this.shape.width * this.shape.height * this.shape.channels); // Input layer expects bytes
        } else {
            this.buffer = ByteBuffer.allocateDirect(this.shape.width * this.shape.height * this.shape.channels * 4); // input layer expects floats
        }
        this.buffer.order(ByteOrder.nativeOrder());

        this.intValues = new int[this.shape.width * this.shape.height];

    }

    //region Getters and Setters

    @Override
    public boolean isQuantized() {
        return quantized;
    }

    @Override
    public ByteBuffer getBackingByteBuffer() {
        return buffer;
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

    private void intPixelToFloat(int pixelValue, ByteBuffer imgData) {
        if (quantized) {
            imgData.put((byte) ((pixelValue >> 16) & 0xFF));
            imgData.put((byte) ((pixelValue >> 8) & 0xFF));
            imgData.put((byte) (pixelValue & 0xFF));
        } else {
            if (this.normalizer != null) {
                imgData.putFloat(this.normalizer.normalize((pixelValue >> 16) & 0xFF, 0));
                imgData.putFloat(this.normalizer.normalize((pixelValue >> 8) & 0xFF, 1));
                imgData.putFloat(this.normalizer.normalize(pixelValue & 0xFF, 2));
            } else {
                imgData.putFloat((pixelValue >> 16) & 0xFF);
                imgData.putFloat((pixelValue >> 8) & 0xFF);
                imgData.putFloat(pixelValue & 0xFF);
            }
        }
    }

    private int floatPixelToInt(float r, float g, float b){
        int rr, gg, bb;
        if (this.denormalizer != null){
            rr = this.denormalizer.denormalize(r, 0);
            gg = this.denormalizer.denormalize(g, 1);
            bb = this.denormalizer.denormalize(b, 2);
        }
        else{
            rr = (int)r;
            gg = (int)g;
            bb = (int)b;
        }
        return 0xFF000000 | (rr << 16) & 0x00FF0000| (gg << 8) & 0x0000FF00 | bb & 0x000000FF;
    }


    @Override
    public ByteBuffer toByteBuffer(Object o) {
        if (o == null) {
            throw new NullPointerException("Input to a model can not be null");
        } else if (!(o instanceof Bitmap)) {
            throw new IllegalArgumentException("Image input should be bitmap");
        }

        Bitmap bitmap = (Bitmap) o;
        if (bitmap.getWidth() != this.shape.width || bitmap.getHeight() != this.shape.height){
            throw new IllegalArgumentException("Image input has the wrong shape, expected width="+this.shape.width+" and height="+this.shape.height+" got width="+bitmap.getWidth()+" and height="+bitmap.getHeight());
        }

        buffer.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < bitmap.getWidth(); ++i) {
            for (int j = 0; j < bitmap.getHeight(); ++j) {
                final int val = intValues[pixel++];
                intPixelToFloat(val, buffer);
            }
        }
        return buffer;
    }

    @Override
    public Bitmap fromByteBuffer(ByteBuffer buffer) {
        buffer.rewind();
        Bitmap bmp = Bitmap.createBitmap(this.shape.width, this.shape.height, Bitmap.Config.ARGB_8888);

        for (int i=0; i<this.shape.width*this.shape.height; i++){
            if (this.quantized){
                int r = buffer.get();
                int g = buffer.get();
                int b = buffer.get();
                this.intValues[i] = 0xFF000000 | (r << 16) & 0x00FF0000| (g << 8) & 0x0000FF00 | b & 0x000000FF;
            }
            else{
                this.intValues[i] = floatPixelToInt(buffer.getFloat(), buffer.getFloat(), buffer.getFloat());
            }
        }

        bmp.setPixels(this.intValues,0, this.shape.width, 0, 0, this.shape.width, this.shape.height);
        return bmp;
    }
}

package ai.doc.tensorio;

import org.junit.Assert;
import org.junit.Test;

import ai.doc.tensorio.TIOData.TIODataDequantizer;
import ai.doc.tensorio.TIOData.TIODataQuantizer;

import static junit.framework.TestCase.assertEquals;

public class QuantizationDequantizationTests {
    @Test
    public void testDataQuantizerStandardZeroToOne() {
        int epsilon = 1;

        TIODataQuantizer quantizer = TIODataQuantizer.TIODataQuantizerZeroToOne();

        Assert.assertEquals(0, quantizer.quantize(0));
        Assert.assertEquals(255, quantizer.quantize(1));
        assertEquals(quantizer.quantize(0.5f), 127, epsilon);
    }

    @Test
    public void testDataQuantizerStandardNegativeOneToOne() {
        float epsilon = 1;
        TIODataQuantizer quantizer = TIODataQuantizer.TIODataQuantizerNegativeOneToOne();

        Assert.assertEquals(0, quantizer.quantize(-1));
        Assert.assertEquals(255, quantizer.quantize(1));
        assertEquals(quantizer.quantize(0), 127, epsilon);
    }

    @Test
    public void testDataQuantizerScaleAndBias() {
        int epsilon = 1;

        TIODataQuantizer quantizer = TIODataQuantizer.TIODataQuantizerWithQuantization(255.0f, 0.0f);
        Assert.assertEquals(0, quantizer.quantize(0));
        Assert.assertEquals(255, quantizer.quantize(1));
        assertEquals(quantizer.quantize(0.5f), 127, epsilon);
    }

    @Test
    public void testDataDequantizerStandardZeroToOne() {
        float epsilon = 0.01f;

        TIODataDequantizer dequantizer = TIODataDequantizer.TIODataDequantizerZeroToOne();
        TIODataQuantizer quantizer = TIODataQuantizer.TIODataQuantizerZeroToOne();
        Assert.assertEquals(0, dequantizer.dequantize(0), 0.0);
        Assert.assertEquals(1, dequantizer.dequantize(255), 0.0);
        assertEquals(dequantizer.dequantize(127), 0.5, epsilon);
    }

    @Test
    public void testDataDequantizerStandardNegativeOneToOne() {
        float epsilon = 0.01f;

        TIODataDequantizer dequantizer = TIODataDequantizer.TIODataDequantizerNegativeOneToOne();

        TIODataQuantizer quantizer = TIODataQuantizer.TIODataQuantizerZeroToOne();
        Assert.assertEquals(dequantizer.dequantize(0), -1, 0.0);
        Assert.assertEquals(1, dequantizer.dequantize(255), 0.0);
        assertEquals(dequantizer.dequantize(127), 0, epsilon);
    }

    @Test
    public void testDataDequantizerScaleAndBias() {
        float epsilon = 0.01f;

        TIODataDequantizer dequantizer = TIODataDequantizer.TIODataDequantizerWithDequantization(1.0f / 255.0f, 0f);

        TIODataQuantizer quantizer = TIODataQuantizer.TIODataQuantizerZeroToOne();
        Assert.assertEquals(0, dequantizer.dequantize(0), 0.0);
        Assert.assertEquals(1, dequantizer.dequantize(255), 0.0);
        assertEquals(dequantizer.dequantize(127), 0.5, epsilon);
    }

}

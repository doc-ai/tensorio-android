/*
 * TIOModelBundleIntegrationTests.java
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

package ai.doc.tensorio;

import android.content.Context;

import org.junit.After;
import org.junit.Before;

import org.junit.Assert;
import org.junit.Test;

import ai.doc.tensorio.core.layerinterface.LayerInterface;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import ai.doc.tensorio.core.modelbundle.ModelBundleException;
import ai.doc.tensorio.core.model.Options;
import ai.doc.tensorio.core.model.PixelFormat;
import androidx.test.platform.app.InstrumentationRegistry;

public class ModelBundleIntegrationTests {
    private Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    private Context testContext = InstrumentationRegistry.getInstrumentation().getContext();

    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testMobileNetModelBundle() {
        float epsilon = 0.01f;

        try {
            ModelBundle bundle = new ModelBundle(testContext, "mobilenet_v2_1.4_224.tiobundle");

            // Basic Properties

            Assert.assertEquals(bundle.getName(), "MobileNet V2 1.0 224");
            Assert.assertEquals(bundle.getDetails(), "MobileNet V2 with a width multiplier of 1.0 and an input resolution of 224x224. \n\nMobileNets are based on a streamlined architecture that have depth-wise separable convolutions to build light weight deep neural networks. Trained on ImageNet with categories such as trees, animals, food, vehicles, person etc. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.");
            Assert.assertEquals(bundle.getIdentifier(), "mobilenet-v2-100-224-unquantized");
            Assert.assertEquals(bundle.getVersion(), "1");
            Assert.assertEquals(bundle.getAuthor(), "Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam");
            Assert.assertEquals(bundle.getLicense(), "Apache License. Version 2.0 http://www.apache.org/licenses/LICENSE-2.0");

            Options options = bundle.getOptions();
            Assert.assertEquals(options.getDevicePosition(), "0");

            // Inputs

            Assert.assertEquals(bundle.getIO().getInputs().size(), 1);
            Assert.assertTrue(bundle.getIO().getInputs().keys().contains("image"));

            LayerInterface input = bundle.getIO().getInputs().get(0);
            Assert.assertEquals(input.getName(), "image");
            Assert.assertSame(input.getMode(), LayerInterface.Mode.Input);

            input.doCase((vectorLayer) -> {
                Assert.fail();
            }, (pixelLayer) -> {
                Assert.assertFalse(pixelLayer.isQuantized());
                Assert.assertSame(pixelLayer.getPixelFormat(), PixelFormat.RGB);

                Assert.assertEquals(pixelLayer.getShape().channels, 3);
                Assert.assertEquals(pixelLayer.getShape().height, 224);
                Assert.assertEquals(pixelLayer.getShape().width, 224);

                Assert.assertEquals(pixelLayer.getNormalizer().normalize(0, 0), -1.0f, epsilon);
                Assert.assertEquals(pixelLayer.getNormalizer().normalize(0, 1), -1.0f, epsilon);
                Assert.assertEquals(pixelLayer.getNormalizer().normalize(0, 2), -1.0f, epsilon);
                Assert.assertEquals(pixelLayer.getNormalizer().normalize(127, 0), 0.0f, epsilon);
                Assert.assertEquals(pixelLayer.getNormalizer().normalize(127, 1), 0.0f, epsilon);
                Assert.assertEquals(pixelLayer.getNormalizer().normalize(127, 2), 0.0f, epsilon);
                Assert.assertEquals(pixelLayer.getNormalizer().normalize(255, 0), 1.0f, epsilon);
                Assert.assertEquals(pixelLayer.getNormalizer().normalize(255, 1), 1.0f, epsilon);
                Assert.assertEquals(pixelLayer.getNormalizer().normalize(255, 2), 1.0f, epsilon);

                Assert.assertNull(pixelLayer.getDenormalizer());
            }, (stringLayer) -> {
                Assert.fail();
            });

            // Outputs

            Assert.assertEquals(bundle.getIO().getOutputs().size(), 1);
            Assert.assertEquals(bundle.getIO().getOutputs().size(), 1);
            Assert.assertTrue(bundle.getIO().getOutputs().keys().contains("classification"));

            LayerInterface output = bundle.getIO().getOutputs().get(0);
            Assert.assertEquals(output.getName(), "classification");
            Assert.assertSame(output.getMode(), LayerInterface.Mode.Output);

            output.doCase((vectorLayer) -> {
                Assert.assertFalse(vectorLayer.isQuantized());
                Assert.assertEquals(vectorLayer.getLength(), 1001);
                Assert.assertTrue(vectorLayer.isLabeled());
                Assert.assertEquals(vectorLayer.getLabels().length, 1001);
                Assert.assertEquals(vectorLayer.getLabels()[0], "background");
                Assert.assertEquals(vectorLayer.getLabels()[vectorLayer.getLabels().length - 1], "toilet tissue");
                Assert.assertNull(vectorLayer.getQuantizer());
                Assert.assertNull(vectorLayer.getDequantizer());
            }, (pixelLayer) -> {
                Assert.fail();
            }, (stringLayer) -> {
                Assert.fail();
            });

        } catch (ModelBundleException e) {
            e.printStackTrace();
            Assert.fail();
        }
    }

    @Test
    public void testMobileNetQuantizedModelBundle() {
        float epsilon = 0.01f;

        try {
            ModelBundle bundle = new ModelBundle(appContext, "mobilenet_v1_1.0_224_quant.tiobundle");

            // Basic Properties

            Assert.assertEquals(bundle.getName(), "MobileNet V1 1.0 224 Quantized");
            Assert.assertEquals(bundle.getDetails(), "MobileNet V1 with a width multiplier of 1.0 and an input resolution of 224x224. Quantized.\n\nMobileNets are based on a streamlined architecture that have depth-wise separable convolutions to build light weight deep neural networks. Trained on ImageNet with categories such as trees, animals, food, vehicles, person etc. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.");
            Assert.assertEquals(bundle.getIdentifier(), "mobilenet-v1-100-224-quantized");
            Assert.assertEquals(bundle.getVersion(), "1");
            Assert.assertEquals(bundle.getAuthor(), "Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam");
            Assert.assertEquals(bundle.getLicense(), "Apache License. Version 2.0 http://www.apache.org/licenses/LICENSE-2.0");

            Options options = bundle.getOptions();
            Assert.assertEquals(options.getDevicePosition(), "0");

            // Inputs

            Assert.assertEquals(bundle.getIO().getInputs().size(), 1);
            Assert.assertTrue(bundle.getIO().getInputs().keys().contains("image"));

            LayerInterface input = bundle.getIO().getInputs().get(0);
            Assert.assertEquals(input.getName(), "image");
            Assert.assertSame(input.getMode(), LayerInterface.Mode.Input);

            input.doCase((vectorLayer) -> {
                Assert.fail();
            }, (pixelLayer) -> {
                Assert.assertTrue(pixelLayer.isQuantized());
                Assert.assertSame(pixelLayer.getPixelFormat(), PixelFormat.RGB);
                Assert.assertEquals(pixelLayer.getShape().channels, 3);
                Assert.assertEquals(pixelLayer.getShape().height, 224);
                Assert.assertEquals(pixelLayer.getShape().width, 224);
                Assert.assertNull(pixelLayer.getNormalizer());
                Assert.assertNull(pixelLayer.getDenormalizer());
            }, (stringLayer) -> {
                Assert.fail();
            });

            // Outputs

            Assert.assertEquals(bundle.getIO().getOutputs().size(), 1);
            Assert.assertEquals(bundle.getIO().getOutputs().size(), 1);
            Assert.assertTrue(bundle.getIO().getOutputs().keys().contains("classification"));

            LayerInterface output = bundle.getIO().getOutputs().get(0);
            Assert.assertEquals(output.getName(), "classification");
            Assert.assertSame(output.getMode(), LayerInterface.Mode.Output);

            output.doCase((vectorLayer) -> {
                Assert.assertTrue(vectorLayer.isQuantized());
                Assert.assertEquals(vectorLayer.getLength(), 1001);
                Assert.assertTrue(vectorLayer.isLabeled());
                Assert.assertEquals(vectorLayer.getLabels().length, 1001);
                Assert.assertEquals(vectorLayer.getLabels()[0], "background");
                Assert.assertEquals(vectorLayer.getLabels()[vectorLayer.getLabels().length - 1], "toilet tissue");
                Assert.assertNull(vectorLayer.getQuantizer());
                Assert.assertNotNull(vectorLayer.getDequantizer());
            }, (pixelLayer) -> {
                Assert.fail();
            }, (stringLayer) -> {
                Assert.fail();
            });

        } catch (ModelBundleException e) {
            e.printStackTrace();
            Assert.fail();
        }
    }
}

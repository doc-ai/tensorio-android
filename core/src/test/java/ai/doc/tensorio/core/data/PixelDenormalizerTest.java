/*
 * TIOPixelDenormalizerTest.java
 * TensorIO
 *
 * Created by Philip Dow on 7/7/2020
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

package ai.doc.tensorio.core.data;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class PixelDenormalizerTest {

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testPixelDenormalizerForDictionaryParsesStandardZeroToOne() {
        PixelDenormalizer denormalizer = PixelDenormalizer.TIOPixelDenormalizerZeroToOne();
        int epsilon = 1;

        assertEquals(denormalizer.denormalize(0.0f, 0), 0, epsilon);
        assertEquals(denormalizer.denormalize(0.0f, 1), 0, epsilon);
        assertEquals(denormalizer.denormalize(0.0f, 2), 0, epsilon);

        assertEquals(denormalizer.denormalize(0.5f, 0), 127, epsilon);
        assertEquals(denormalizer.denormalize(0.5f, 1), 127, epsilon);
        assertEquals(denormalizer.denormalize(0.5f, 2), 127, epsilon);

        assertEquals(denormalizer.denormalize(1.0f, 0), 255, epsilon);
        assertEquals(denormalizer.denormalize(1.0f, 1), 255, epsilon);
        assertEquals(denormalizer.denormalize(1.0f, 2), 255, epsilon);
    }

    @Test
    public void testPixelDenormalizerForDictionaryParsesStandardNegativeOneToOne() {
        PixelDenormalizer denormalizer = PixelDenormalizer.TIOPixelDenormalizerNegativeOneToOne();
        int epsilon = 1;

        assertEquals(denormalizer.denormalize(-1.0f, 0), 0, epsilon);
        assertEquals(denormalizer.denormalize(-1.0f, 1), 0, epsilon);
        assertEquals(denormalizer.denormalize(-1.0f, 2), 0, epsilon);

        assertEquals(denormalizer.denormalize(0.0f, 0), 127, epsilon);
        assertEquals(denormalizer.denormalize(0.0f, 1), 127, epsilon);
        assertEquals(denormalizer.denormalize(0.0f, 2), 127, epsilon);

        assertEquals(denormalizer.denormalize(1.0f, 0), 255, epsilon);
        assertEquals(denormalizer.denormalize(1.0f, 1), 255, epsilon);
        assertEquals(denormalizer.denormalize(1.0f, 2), 255, epsilon);
    }

    @Test
    public void testPixelDenormalizerForDictionaryParsesScaleAndSameBiases() {
        PixelDenormalizer denormalizer = PixelDenormalizer.TIOPixelDenormalizerSingleBias(255.0f, 0.0f);
        int epsilon = 1;

        assertEquals(denormalizer.denormalize(0.0f, 0), 0, epsilon);
        assertEquals(denormalizer.denormalize(0.0f, 1), 0, epsilon);
        assertEquals(denormalizer.denormalize(0.0f, 2), 0, epsilon);

        assertEquals(denormalizer.denormalize(0.5f, 0), 127, epsilon);
        assertEquals(denormalizer.denormalize(0.5f, 1), 127, epsilon);
        assertEquals(denormalizer.denormalize(0.5f, 2), 127, epsilon);

        assertEquals(denormalizer.denormalize(1.0f, 0), 255, epsilon);
        assertEquals(denormalizer.denormalize(1.0f, 1), 255, epsilon);
        assertEquals(denormalizer.denormalize(1.0f, 2), 255, epsilon);
    }

    @Test
    public void testPixelDenormalizerForDictionaryParsesScaleAndDifferenceBiases() {
        PixelDenormalizer denormalizer = PixelDenormalizer.TIOPixelDenormalizerPerChannelBias(255.0f, -0.1f, -0.2f, -0.3f);
        int epsilon = 1;

        assertEquals(denormalizer.denormalize(0.0f + 0.1f, 0), 0, epsilon);
        assertEquals(denormalizer.denormalize(0.0f + 0.2f, 1), 0, epsilon);
        assertEquals(denormalizer.denormalize(0.0f + 0.3f, 2), 0, epsilon);

        assertEquals(denormalizer.denormalize(0.5f + 0.1f, 0), 127, epsilon);
        assertEquals(denormalizer.denormalize(0.5f + 0.2f, 1), 127, epsilon);
        assertEquals(denormalizer.denormalize(0.5f + 0.3f, 2), 127, epsilon);

        assertEquals(denormalizer.denormalize(1.0f + 0.1f, 0), 255, epsilon);
        assertEquals(denormalizer.denormalize(1.0f + 0.2f, 1), 255, epsilon);
        assertEquals(denormalizer.denormalize(1.0f + 0.3f, 2), 255, epsilon);
    }
}
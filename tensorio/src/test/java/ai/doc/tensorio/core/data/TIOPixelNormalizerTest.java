/*
 * TIOPixelNormalizerTest.java
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

public class TIOPixelNormalizerTest {

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testPixelNormalizerStandardZeroToOne() {
        TIOPixelNormalizer normalizer = TIOPixelNormalizer.TIOPixelNormalizerZeroToOne();
        float epsilon = 0.01f;

        assertEquals(normalizer.normalize(0, 0), 0.0, epsilon);
        assertEquals(normalizer.normalize(0, 1), 0.0, epsilon);
        assertEquals(normalizer.normalize(0, 2), 0.0, epsilon);

        assertEquals(normalizer.normalize(127, 0), 0.5, epsilon);
        assertEquals(normalizer.normalize(127, 1), 0.5, epsilon);
        assertEquals(normalizer.normalize(127, 2), 0.5, epsilon);

        assertEquals(normalizer.normalize(255, 0), 1.0, epsilon);
        assertEquals(normalizer.normalize(255, 1), 1.0, epsilon);
        assertEquals(normalizer.normalize(255, 2), 1.0, epsilon);
    }

    @Test
    public void testPixelNormalizerStandardNegativeOneToOne() {
        TIOPixelNormalizer normalizer = TIOPixelNormalizer.TIOPixelNormalizerNegativeOneToOne();
        float epsilon = 0.01f;

        assertEquals(normalizer.normalize(0, 0), -1.0, epsilon);
        assertEquals(normalizer.normalize(0, 1), -1.0, epsilon);
        assertEquals(normalizer.normalize(0, 2), -1.0, epsilon);

        assertEquals(normalizer.normalize(127, 0), 0.0, epsilon);
        assertEquals(normalizer.normalize(127, 1), 0.0, epsilon);
        assertEquals(normalizer.normalize(127, 2), 0.0, epsilon);

        assertEquals(normalizer.normalize(255, 0), 1.0, epsilon);
        assertEquals(normalizer.normalize(255, 1), 1.0, epsilon);
        assertEquals(normalizer.normalize(255, 2), 1.0, epsilon);
    }

    @Test
    public void testPixelNormalizerScaleAndSameBiases() {
        TIOPixelNormalizer normalizer = TIOPixelNormalizer.TIOPixelNormalizerSingleBias(1.0f / 255.0f, 0.0f);
        float epsilon = 0.01f;

        assertEquals(normalizer.normalize(0, 0), 0.0, epsilon);
        assertEquals(normalizer.normalize(0, 1), 0.0, epsilon);
        assertEquals(normalizer.normalize(0, 2), 0.0, epsilon);

        assertEquals(normalizer.normalize(127, 0), 0.5, epsilon);
        assertEquals(normalizer.normalize(127, 1), 0.5, epsilon);
        assertEquals(normalizer.normalize(127, 2), 0.5, epsilon);

        assertEquals(normalizer.normalize(255, 0), 1.0, epsilon);
        assertEquals(normalizer.normalize(255, 1), 1.0, epsilon);
        assertEquals(normalizer.normalize(255, 2), 1.0, epsilon);
    }

    @Test
    public void testPixelNormalizerScaleAndDifferenceBiases() {
        TIOPixelNormalizer normalizer = TIOPixelNormalizer.TIOPixelNormalizerPerChannelBias(1.0f / 255.0f, 0.1f, 0.2f, 0.3f);
        float epsilon = 0.01f;

        assertEquals(normalizer.normalize(0, 0), 0.0 + 0.1, epsilon);
        assertEquals(normalizer.normalize(0, 1), 0.0 + 0.2, epsilon);
        assertEquals(normalizer.normalize(0, 2), 0.0 + 0.3, epsilon);

        assertEquals(normalizer.normalize(127, 0), 0.5 + 0.1, epsilon);
        assertEquals(normalizer.normalize(127, 1), 0.5 + 0.2, epsilon);
        assertEquals(normalizer.normalize(127, 2), 0.5 + 0.3, epsilon);

        assertEquals(normalizer.normalize(255, 0), 1.0 + 0.1, epsilon);
        assertEquals(normalizer.normalize(255, 1), 1.0 + 0.2, epsilon);
        assertEquals(normalizer.normalize(255, 2), 1.0 + 0.3, epsilon);
    }
}
/*
 * TIODataQuantizerTest.java
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

package ai.doc.tensorio.core.TIOData;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class TIODataQuantizerTest {

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testDataQuantizerStandardZeroToOne() {
        TIODataQuantizer quantizer = TIODataQuantizer.TIODataQuantizerZeroToOne();
        int epsilon = 1;

        assertEquals(0, quantizer.quantize(0), epsilon);
        assertEquals(255, quantizer.quantize(1), epsilon);
        assertEquals(127, quantizer.quantize(0.5f), epsilon);
    }

    @Test
    public void testDataQuantizerStandardNegativeOneToOne() {
        TIODataQuantizer quantizer = TIODataQuantizer.TIODataQuantizerNegativeOneToOne();
        int epsilon = 1;

        assertEquals(0, quantizer.quantize(-1), epsilon);
        assertEquals(255, quantizer.quantize(1), epsilon);
        assertEquals(127, quantizer.quantize(0), epsilon);
    }

    @Test
    public void testDataQuantizerScaleAndBias() {
        TIODataQuantizer quantizer = TIODataQuantizer.TIODataQuantizerWithQuantization(255.0f, 0.0f);
        int epsilon = 1;

        assertEquals(0, quantizer.quantize(0), epsilon);
        assertEquals(255, quantizer.quantize(1), epsilon);
        assertEquals(127, quantizer.quantize(0.5f), epsilon);
    }
}
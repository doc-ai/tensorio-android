/*
 * TIODataDequantizerTest.java
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

import static junit.framework.TestCase.assertEquals;

public class TIODataDequantizerTest {

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testDataDequantizerStandardZeroToOne() {
        TIODataDequantizer dequantizer = TIODataDequantizer.TIODataDequantizerZeroToOne();
        float epsilon = 0.01f;

        assertEquals(0, dequantizer.dequantize(0), epsilon);
        assertEquals(1, dequantizer.dequantize(255), epsilon);
        assertEquals(0.5, dequantizer.dequantize(127), epsilon);
    }

    @Test
    public void testDataDequantizerStandardNegativeOneToOne() {
        TIODataDequantizer dequantizer = TIODataDequantizer.TIODataDequantizerNegativeOneToOne();
        float epsilon = 0.01f;

        assertEquals(dequantizer.dequantize(0), -1, epsilon);
        assertEquals(1, dequantizer.dequantize(255), epsilon);
        assertEquals(0, dequantizer.dequantize(127), epsilon);
    }

    @Test
    public void testDataDequantizerScaleAndBias() {
        TIODataDequantizer dequantizer = TIODataDequantizer.TIODataDequantizerWithDequantization(1.0f / 255.0f, 0f);
        float epsilon = 0.01f;

        assertEquals(0, dequantizer.dequantize(0), epsilon);
        assertEquals(1, dequantizer.dequantize(255), epsilon);
        assertEquals(0.5, dequantizer.dequantize(127), epsilon);
    }
}
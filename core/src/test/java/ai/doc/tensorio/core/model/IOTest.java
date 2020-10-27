/*
 * IOTest.java
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

package ai.doc.tensorio.core.model;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import ai.doc.tensorio.core.layerinterface.DataType;
import ai.doc.tensorio.core.layerinterface.LayerInterface;
import ai.doc.tensorio.core.layerinterface.VectorLayerDescription;

import static junit.framework.TestCase.assertEquals;

public class IOTest {

    private LayerInterface fooIn;
    private LayerInterface barIn;

    private LayerInterface fooOut;
    private LayerInterface barOut;

    private LayerInterface fooPlaceholder;
    private LayerInterface barPlaceholder;

    @Before
    public void setUp() throws Exception {
        this.fooIn =  new LayerInterface("foo", LayerInterface.Mode.Input, new VectorLayerDescription(
                new int[]{1},
                false,
                null,
                false,
                null,
                null,
                DataType.Float32
        ));
        this.barIn =  new LayerInterface("bar", LayerInterface.Mode.Input, new VectorLayerDescription(
                new int[]{1},
                false,
                null,
                false,
                null,
                null,
                DataType.Float32
        ));

        this.fooOut =  new LayerInterface("foo", LayerInterface.Mode.Output, new VectorLayerDescription(
                new int[]{1},
                false,
                null,
                false,
                null,
                null,
                DataType.Float32
        ));
        this.barOut =  new LayerInterface("bar", LayerInterface.Mode.Output, new VectorLayerDescription(
                new int[]{1},
                false,
                null,
                false,
                null,
                null,
                DataType.Float32
        ));

        this.fooPlaceholder =  new LayerInterface("foo", LayerInterface.Mode.Placeholder, new VectorLayerDescription(
                new int[]{1},
                false,
                null,
                false,
                null,
                null,
                DataType.Float32
        ));
        this.barPlaceholder =  new LayerInterface("bar", LayerInterface.Mode.Placeholder, new VectorLayerDescription(
                new int[]{1},
                false,
                null,
                false,
                null,
                null,
                DataType.Float32
        ));
    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testModelIOPreservesIndex() {
        IO io = new IO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().get(0), this.fooIn);
        assertEquals(io.getInputs().get(1), this.barIn);

        assertEquals(io.getOutputs().get(0), this.fooOut);
        assertEquals(io.getOutputs().get(1), this.barOut);

        assertEquals(io.getPlaceholders().get(0), this.fooPlaceholder);
        assertEquals(io.getPlaceholders().get(1), this.barPlaceholder);
    }

    @Test
    public void testModelIOPreservesName() {
        IO io = new IO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().get("foo"), this.fooIn);
        assertEquals(io.getInputs().get("bar"), this.barIn);

        assertEquals(io.getOutputs().get("foo"), this.fooOut);
        assertEquals(io.getOutputs().get("bar"), this.barOut);

        assertEquals(io.getPlaceholders().get("foo"), this.fooPlaceholder);
        assertEquals(io.getPlaceholders().get("bar"), this.barPlaceholder);
    }

    @Test
    public void testModelIOReturnsAllObjects() {
        IO io = new IO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().all(), Arrays.asList(this.fooIn, this.barIn));
        assertEquals(io.getOutputs().all(), Arrays.asList(this.fooOut, this.barOut));
        assertEquals(io.getPlaceholders().all(), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));
    }

    @Test
    public void testModelIOReturnsAllKeys() {
        IO io = new IO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));
        Set keys = new HashSet<String>(Arrays.asList(new String[]{"foo", "bar"}));

        assertEquals(io.getInputs().keys(), keys);
        assertEquals(io.getOutputs().keys(), keys);
        assertEquals(io.getPlaceholders().keys(), keys);
    }

    @Test
    public void testModelIOCountIsCorrect() {
        IO io = new IO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().size(), 2);
        assertEquals(io.getOutputs().size(), 2);
        assertEquals(io.getPlaceholders().size(), 2);
    }

    @Test
    public void testModelIOReturnsIndexForName() {
        IO io = new IO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().indexFor("foo"), Integer.valueOf(0));
        assertEquals(io.getInputs().indexFor("bar"), Integer.valueOf(1));

        assertEquals(io.getOutputs().indexFor("foo"), Integer.valueOf(0));
        assertEquals(io.getOutputs().indexFor("bar"), Integer.valueOf(1));

        assertEquals(io.getPlaceholders().indexFor("foo"), Integer.valueOf(0));
        assertEquals(io.getPlaceholders().indexFor("bar"), Integer.valueOf(1));
    }
}

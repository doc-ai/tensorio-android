/*
 * TIOModelIOTest.java
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

package ai.doc.tensorio.TIOModel;

import com.google.common.collect.Sets;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;
import ai.doc.tensorio.TIOLayerInterface.TIOVectorLayerDescription;

import static junit.framework.TestCase.assertEquals;

public class TIOModelIOTest {

    private TIOLayerInterface fooIn;
    private TIOLayerInterface barIn;

    private TIOLayerInterface fooOut;
    private TIOLayerInterface barOut;

    private TIOLayerInterface fooPlaceholder;
    private TIOLayerInterface barPlaceholder;

    @Before
    public void setUp() throws Exception {
        this.fooIn =  new TIOLayerInterface("foo", TIOLayerInterface.Mode.Input, new TIOVectorLayerDescription(
                new int[]{1},
                null,
                false,
                null,
                null
        ));
        this.barIn =  new TIOLayerInterface("bar", TIOLayerInterface.Mode.Input, new TIOVectorLayerDescription(
                new int[]{1},
                null,
                false,
                null,
                null
        ));

        this.fooOut =  new TIOLayerInterface("foo", TIOLayerInterface.Mode.Output, new TIOVectorLayerDescription(
                new int[]{1},
                null,
                false,
                null,
                null
        ));
        this.barOut =  new TIOLayerInterface("bar", TIOLayerInterface.Mode.Output, new TIOVectorLayerDescription(
                new int[]{1},
                null,
                false,
                null,
                null
        ));

        this.fooPlaceholder =  new TIOLayerInterface("foo", TIOLayerInterface.Mode.Placeholder, new TIOVectorLayerDescription(
                new int[]{1},
                null,
                false,
                null,
                null
        ));
        this.barPlaceholder =  new TIOLayerInterface("bar", TIOLayerInterface.Mode.Placeholder, new TIOVectorLayerDescription(
                new int[]{1},
                null,
                false,
                null,
                null
        ));
    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testModelIOPreservesIndex() {
        TIOModelIO io = new TIOModelIO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().get(0), this.fooIn);
        assertEquals(io.getInputs().get(1), this.barIn);

        assertEquals(io.getOutputs().get(0), this.fooOut);
        assertEquals(io.getOutputs().get(1), this.barOut);

        assertEquals(io.getPlaceholders().get(0), this.fooPlaceholder);
        assertEquals(io.getPlaceholders().get(1), this.barPlaceholder);
    }

    @Test
    public void testModelIOPreservesName() {
        TIOModelIO io = new TIOModelIO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().get("foo"), this.fooIn);
        assertEquals(io.getInputs().get("bar"), this.barIn);

        assertEquals(io.getOutputs().get("foo"), this.fooOut);
        assertEquals(io.getOutputs().get("bar"), this.barOut);

        assertEquals(io.getPlaceholders().get("foo"), this.fooPlaceholder);
        assertEquals(io.getPlaceholders().get("bar"), this.barPlaceholder);
    }

    @Test
    public void testModelIOReturnsAllObjects() {
        TIOModelIO io = new TIOModelIO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().all(), Arrays.asList(this.fooIn, this.barIn));
        assertEquals(io.getOutputs().all(), Arrays.asList(this.fooOut, this.barOut));
        assertEquals(io.getPlaceholders().all(), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));
    }

    @Test
    public void testModelIOReturnsAllKeys() {
        TIOModelIO io = new TIOModelIO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().keys(), Sets.newHashSet("foo", "bar"));
        assertEquals(io.getOutputs().keys(), Sets.newHashSet("foo", "bar"));
        assertEquals(io.getPlaceholders().keys(), Sets.newHashSet("foo", "bar"));
    }

    @Test
    public void testModelIOCountIsCorrect() {
        TIOModelIO io = new TIOModelIO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().size(), 2);
        assertEquals(io.getOutputs().size(), 2);
        assertEquals(io.getPlaceholders().size(), 2);
    }

    @Test
    public void testModelIOReturnsIndexForName() {
        TIOModelIO io = new TIOModelIO(Arrays.asList(this.fooIn, this.barIn), Arrays.asList(this.fooOut, this.barOut), Arrays.asList(this.fooPlaceholder, this.barPlaceholder));

        assertEquals(io.getInputs().indexFor("foo"), Integer.valueOf(0));
        assertEquals(io.getInputs().indexFor("bar"), Integer.valueOf(1));

        assertEquals(io.getOutputs().indexFor("foo"), Integer.valueOf(0));
        assertEquals(io.getOutputs().indexFor("bar"), Integer.valueOf(1));

        assertEquals(io.getPlaceholders().indexFor("foo"), Integer.valueOf(0));
        assertEquals(io.getPlaceholders().indexFor("bar"), Integer.valueOf(1));
    }
}

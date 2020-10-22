package ai.doc.tensorio.core.training;

import org.junit.After;
import org.junit.Before;

import ai.doc.tensorio.core.data.Batch;
import ai.doc.tensorio.core.data.InMemoryBatchDataSource;

import static org.junit.Assert.*;

// TODO: Implement but use mocks for it

public class ModelTrainerTest {

    private Batch.Item item1;
    private Batch.Item item2;
    private Batch.Item item3;

    @Before
    public void setUp() throws Exception {
        createItem1();
        createItem2();
        createItem3();
    }

    @After
    public void tearDown() throws Exception {
        item1 = null;
        item2 = null;
        item3 = null;
    }

    private void createItem1() {
        float[] input = {
                1,2,3,4
        };
        float[] label = {
                0
        };

        item1 = new Batch.Item();
        item1.put("input", input);
        item1.put("label", label);
    }

    private void createItem2() {
        float[] input = {
                5,6,7,8
        };
        float[] label = {
                1
        };

        item2 = new Batch.Item();
        item2.put("input", input);
        item2.put("label", label);
    }

    private void createItem3() {
        float[] input = {
                2,4,6,8
        };
        float[] label = {
                2
        };

        item3 = new Batch.Item();
        item3.put("input", input);
        item3.put("label", label);
    }


}
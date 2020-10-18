package ai.doc.tensorio.core.data;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class BatchTest {

    static float epsilon = (float) 0.01;

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testEmptyBatchWithKeys() {
        String[] keys = {"input", "label"};
        Batch batch = new Batch(keys);

        assertEquals(2, batch.getKeys().length);
        assertTrue(Arrays.asList(batch.getKeys()).contains("input"));
        assertTrue(Arrays.asList(batch.getKeys()).contains("label"));
    }

    @Test
    public void testBatchWithItem() {
        Batch.Item item = new Batch.Item();
        item.put("input", new float[]{1});
        item.put("label", new float[]{2});

        Batch batch = new Batch(item);

        assertEquals(2, batch.getKeys().length);
        assertTrue(Arrays.asList(batch.getKeys()).contains("input"));
        assertTrue(Arrays.asList(batch.getKeys()).contains("label"));

        assertEquals(1, batch.size());
        assertEquals(item, batch.get(0));

        Object[] inputs = batch.valuesForKey("input");
        Object[] labels = batch.valuesForKey("label");

        assertEquals(1, inputs.length);
        assertArrayEquals((float[])inputs[0], new float[]{1}, epsilon);

        assertEquals(1, labels.length);
        assertArrayEquals((float[])labels[0], new float[]{2}, epsilon);
    }

    @Test
    public void testBatchWithItems() {
        try {
            Batch.Item item1 = new Batch.Item();
            item1.put("input", new float[]{1,2,3,4});
            item1.put("label", new float[]{2});

            Batch.Item item2 = new Batch.Item();
            item2.put("input", new float[]{10,20,30,40});
            item2.put("label", new float[]{20});

            Batch.Item[] items = {item1, item2};
            Batch batch = new Batch(items);

            assertEquals(2, batch.getKeys().length);
            assertTrue(Arrays.asList(batch.getKeys()).contains("input"));
            assertTrue(Arrays.asList(batch.getKeys()).contains("label"));

            Batch.Item r1 = batch.get(0);
            Batch.Item r2 = batch.get(1);

            assertEquals(2, batch.size());
            assertEquals(item1, batch.get(0));
            assertEquals(item2, batch.get(1));

            Object[] inputs = batch.valuesForKey("input");
            Object[] labels = batch.valuesForKey("label");

            assertEquals(2, inputs.length);
            assertArrayEquals((float[]) inputs[0], new float[]{1, 2, 3, 4}, epsilon);
            assertArrayEquals((float[]) inputs[1], new float[]{10, 20, 30, 40}, epsilon);

            assertEquals(2, labels.length);
            assertArrayEquals((float[]) labels[0], new float[]{2}, epsilon);
            assertArrayEquals((float[]) labels[1], new float[]{20}, epsilon);

        } catch (IllegalArgumentException e) {
            fail();
        }
    }

    @Test
    public void testBatchWithItemsDifferentKeysThrowsException() {
        try {
            Batch.Item item1 = new Batch.Item();
            item1.put("foo", new float[]{1, 2, 3, 4});
            item1.put("bar", new float[]{2});

            Batch.Item item2 = new Batch.Item();
            item2.put("fuz", new float[]{10, 20, 30, 40});
            item2.put("baz", new float[]{20});

            Batch.Item[] items = {item1, item2};
            Batch batch = new Batch(items);

            fail();

        } catch (IllegalArgumentException e) {
            assertTrue(true);
        }
    }

    @Test
    public void testAddsItemToEmptyBatch() {
        String[] keys = {"input", "label"};
        Batch batch = new Batch(keys);

        Batch.Item item = new Batch.Item();
        item.put("input", new float[]{1});
        item.put("label", new float[]{2});

        batch.add(item);

        assertEquals(2, batch.getKeys().length);
        assertTrue(Arrays.asList(batch.getKeys()).contains("input"));
        assertTrue(Arrays.asList(batch.getKeys()).contains("label"));

        assertEquals(1, batch.size());
        assertEquals(item, batch.get(0));

        Object[] inputs = batch.valuesForKey("input");
        Object[] labels = batch.valuesForKey("label");

        assertEquals(1, inputs.length);
        assertArrayEquals((float[])inputs[0], new float[]{1}, epsilon);

        assertEquals(1, labels.length);
        assertArrayEquals((float[])labels[0], new float[]{2}, epsilon);
    }

    @Test
    public void testAddsItemToNonEmptyBatch() {
        Batch.Item item1 = new Batch.Item();
        item1.put("input", new float[]{1,2,3,4});
        item1.put("label", new float[]{2});

        Batch.Item item2 = new Batch.Item();
        item2.put("input", new float[]{10,20,30,40});
        item2.put("label", new float[]{20});

        Batch batch = new Batch(item1);
        batch.add(item2);

        assertEquals(2, batch.getKeys().length);
        assertTrue(Arrays.asList(batch.getKeys()).contains("input"));
        assertTrue(Arrays.asList(batch.getKeys()).contains("label"));

        Batch.Item r1 = batch.get(0);
        Batch.Item r2 = batch.get(1);

        assertEquals(2, batch.size());
        assertEquals(item1, batch.get(0));
        assertEquals(item2, batch.get(1));

        Object[] inputs = batch.valuesForKey("input");
        Object[] labels = batch.valuesForKey("label");

        assertEquals(2, inputs.length);
        assertArrayEquals((float[]) inputs[0], new float[]{1, 2, 3, 4}, epsilon);
        assertArrayEquals((float[]) inputs[1], new float[]{10, 20, 30, 40}, epsilon);

        assertEquals(2, labels.length);
        assertArrayEquals((float[]) labels[0], new float[]{2}, epsilon);
        assertArrayEquals((float[]) labels[1], new float[]{20}, epsilon);
    }

    @Test
    public void testAddsItemToBatchWithDifferentKeysThrowsException() {
        try {
            Batch.Item item1 = new Batch.Item();
            item1.put("foo", new float[]{1, 2, 3, 4});
            item1.put("bar", new float[]{2});

            Batch.Item item2 = new Batch.Item();
            item2.put("fuz", new float[]{10, 20, 30, 40});
            item2.put("baz", new float[]{20});

            Batch batch = new Batch(item1);
            batch.add(item2);

            fail();

        } catch (IllegalArgumentException e) {
            assertTrue(true);
        }
    }
}
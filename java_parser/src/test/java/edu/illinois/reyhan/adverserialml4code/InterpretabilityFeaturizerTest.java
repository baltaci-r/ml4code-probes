package edu.illinois.reyhan.adverserialml4code;

import org.junit.jupiter.api.Test;

import edu.illinois.reyhan.ml4code.InterpretabilityFeaturizer;
import spoon.Launcher;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.visitor.Filter;

public class InterpretabilityFeaturizerTest {
    @Test
    public void test1() {
        var method = Launcher.parseClass("""
            public class FooTest {
                public void test() {
                    int i = 0;
                    while (i < 10) {
                        i++;
                    }
                }
            }
        """).getElements(new Filter<CtMethod<?>>() {
            @Override
            public boolean matches(CtMethod<?> element) {
                return true;
            }
        }).get(0);

        var featurizer = new InterpretabilityFeaturizer(method);
        assert(featurizer.hasWhileStatement());
        assert(!featurizer.hasForLoop());
        assert(!featurizer.hasSwitchStatement());
        assert(!featurizer.hasIfStatement());
    }

    @Test
    public void test2() {
        var method = Launcher.parseClass("""
            public class FooTest {
                public void test() {
                    int i = 0;
                    while (i < 10) {
                        i++;
                    }
                    for (int j = 0; j < 10; j++) {
                        i++;
                    }
                }
            }
        """).getElements(new Filter<CtMethod<?>>() {
            @Override
            public boolean matches(CtMethod<?> element) {
                return true;
            }
        }).get(0);

        var featurizer = new InterpretabilityFeaturizer(method);
        assert(featurizer.hasWhileStatement());
        assert(featurizer.hasForLoop());
        assert(!featurizer.hasSwitchStatement());
        assert(!featurizer.hasIfStatement());
    }

    @Test
    public void test3() {
        var method = Launcher.parseClass("""
            public class FooTest {
                public void test() {
                    int i = 0;
                    while (i < 10) {
                        i++;
                    }
                    if (i < 10) {
                        i++;
                    }
                }
            }
        """).getElements(new Filter<CtMethod<?>>() {
            @Override
            public boolean matches(CtMethod<?> element) {
                return true;
            }
        }).get(0);

        var featurizer = new InterpretabilityFeaturizer(method);
        assert(featurizer.hasWhileStatement());
        assert(!featurizer.hasForLoop());
        assert(!featurizer.hasSwitchStatement());
        assert(featurizer.hasIfStatement());
    }
}

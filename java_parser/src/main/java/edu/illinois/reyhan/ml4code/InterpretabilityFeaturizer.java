package edu.illinois.reyhan.ml4code;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.Dictionary;
import java.util.Map;

import com.fasterxml.jackson.core.JsonFactory;

import spoon.reflect.code.CtCatch;
import spoon.reflect.code.CtFor;
import spoon.reflect.code.CtForEach;
import spoon.reflect.code.CtIf;
import spoon.reflect.code.CtInvocation;
import spoon.reflect.code.CtLoop;
import spoon.reflect.code.CtSwitch;
import spoon.reflect.code.CtWhile;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.visitor.Filter;

public class InterpretabilityFeaturizer {
    // takes a method gives a list of features

    public CtMethod<?> method;

    public InterpretabilityFeaturizer(CtMethod<?> method) {
        this.method = method;
    }

    public Boolean hasSwitchStatement() {
        return method.getElements(new Filter<CtSwitch<?>>() {
            @Override
            public boolean matches(CtSwitch<?> element) {
                return true;
            }
        }).size() > 0;
    }

    public Boolean hasIfStatement() {
        return method.getElements(new Filter<CtIf>() {
            @Override
            public boolean matches(CtIf element) {
                return true;
            }
        }).size() > 0;
    }

    public Boolean hasWhileStatement() {
        return method.getElements(new Filter<CtWhile>() {
            @Override
            public boolean matches(CtWhile element) {
                return true;
            }
        }).size() > 0;
    }

    public Boolean hasForLoop() {
        return method.getElements(new Filter<CtLoop>() {
            @Override
            public boolean matches(CtLoop element) {
                return element instanceof CtFor || element instanceof CtForEach;
            }
        }).size() > 0;
    }

    public Boolean hasFunctionCalls() {
        return method.getElements(new Filter<CtInvocation<?>>() {
            @Override
            public boolean matches(CtInvocation<?> element) {
                return true;
            }
        }).size() > 0;
    }

    public Boolean throwsException() {
        return method.getThrownTypes().size() > 0;
    }

    public Boolean catchesException() {
        return method.getElements(new Filter<CtCatch>() {
            @Override
            public boolean matches(CtCatch element) {
                return true;
            }
        }).size() > 0;
    }

    public Map<String, Boolean> getFeatures() {
        Map<String, Boolean> features = new java.util.Hashtable<String, Boolean>();
        features.put("hasSwitchStatement", hasSwitchStatement());
        features.put("hasIfStatement", hasIfStatement());
        features.put("hasWhileStatement", hasWhileStatement());
        features.put("hasForLoop", hasForLoop());
        features.put("hasFunctionCalls", hasFunctionCalls());
        features.put("throwsException", throwsException());
        features.put("catchesException", catchesException());
        return features;
    }

    public void toJson(PrintStream out) throws Exception {
        // { "class": "A", "method": "m", code: "<code>", "f1": 1, "f2": 2, ... }
        var json = new JsonFactory();
        var tmpOut = new ByteArrayOutputStream();
        var mapper = json.createGenerator(tmpOut);
        mapper.writeStartObject();
        mapper.writeStringField("class", method.getDeclaringType().getSimpleName());
        mapper.writeStringField("method", method.getSimpleName());
        mapper.writeStringField("code", method.toString());
        // mapper.writeFieldName("features");
        // mapper.writeStartObject();
        for (var key : getFeatures().keySet()) {
            mapper.writeStringField(key, getFeatures().get(key).toString());
        }
        // mapper.writeEndObject();
        mapper.writeEndObject();
        mapper.close();

        out.println(tmpOut.toString());
    }
}

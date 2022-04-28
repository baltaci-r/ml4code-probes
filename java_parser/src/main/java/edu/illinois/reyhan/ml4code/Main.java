package edu.illinois.reyhan.ml4code;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import spoon.Launcher;

public class Main {

    
    public Main() {
        
    }

    public static void main(String[] args) throws Exception {
        var M = new Main();

        ArgumentParser parser = ArgumentParsers.newFor("JavaParser").build()
                .defaultHelp(true)
                .description("Parse Java code and extract features");
        
        // common options
        parser.addArgument("-f", "--format").choices("method", "class", "java_file").setDefault("method").help("Format of input. Can be 'method', 'class' or 'java_file'. First two must be supplied via stdin.");
        parser.addArgument("-i", "--input").help("Input file").required(false);
        parser.addArgument("-d", "--dir").help("Input directory").required(false);
        parser.addArgument("-o", "--output").help("Output file (only if -i is used)").required(false);

        // one of input or dir must be specified.

        Namespace ns = null;
        try {
            ns = parser.parseArgs(args);

            if (ns.getString("format").equals("java_file") && (
                       ns.getString("input") == null)
                    && (ns.getString("dir") == null)
                ) {
                M.error("-i or -d must be specified if -f is java_file");
            }

        } catch (ArgumentParserException e1) {
            parser.handleError(e1);
            System.exit(1);
        }

        M.run(ns);
    }

    private void error(String string) {
        System.err.println(string);
        System.exit(-1);
    }

    public ArrayList<InterpretabilityFeaturizer> getFeaturizers(String code, Namespace ns) throws Exception {
        if (ns.getString("format") == "method") {
            code = "class A { " + code + " }";
        }

        var cls = Launcher.parseClass(code);
        if (cls.getMethods().size() == 0) {
            throw new Exception("No method found in the code");
        }
        
        var featurizers = new ArrayList<InterpretabilityFeaturizer>();
        for (var m : cls.getMethods()) {
            featurizers.add(new InterpretabilityFeaturizer(m));
        }
        return featurizers;
    }

    public void lineWiseFeaturizer(String inputFileName, Namespace ns, PrintStream out) throws Exception {
        // each line in inputFile is a class
        var reader = new BufferedReader(new FileReader(inputFileName));
        String line;
        int i = 0;
        
        while ((line = reader.readLine()) != null) {
            ++i;
            try {
                var featurizers = getFeaturizers(line, ns);
                for (var f : featurizers) {
                    f.toJson(out);
                }
            } catch (Exception e) {
                System.err.println("Error in processing line#" + i + "\t" + e);
            }
        }
        
        reader.close();
    }

    public void fileFeaturizer(String inputFileName, Namespace ns, PrintStream out) throws Exception {
        // inputFilePath is path of a .java file
        var reader = new BufferedReader(new FileReader(inputFileName));

        // read all lines
        String line;
        StringBuilder sb = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            sb.append(line);
            sb.append("\n");
        }
        
        reader.close();

        var featurizers = getFeaturizers(sb.toString(), ns);
        for (var f : featurizers) {
            f.toJson(out);
        }
    }

    public void dirFeaturizer(String inputDirName, Namespace ns, PrintStream out) throws Exception {
        // inputDirPath is path of a directory
        
        // get all files in the directory (recursively)
        var files = Files.walk(Paths.get(inputDirName))
                .filter(Files::isRegularFile)
                .map(Path::toString)
                .collect(Collectors.toList());
        
        // process files in parallel
        var pool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        var futures = new ArrayList<Future<Object>>();
        for (var file : files) {
            var f = pool.submit(() -> {
                try {
                    fileFeaturizer(file, ns, out);
                } catch (Exception e) {
                    if (!e.toString().contains("parseClass only considers") && !e.toString().contains("No method found in the code")) {
                        System.err.println("Error in processing file " + file + "\t" + e);
                    }
                }
                return null;
            });
            futures.add(f);
        }

        // wait for all files to be processed
        int i = 0;
        long step = Math.round(files.size() * 0.05);
        for (var f : futures) {
            try {
                f.get(10, TimeUnit.SECONDS);
                if (i % step == 0) {
                    System.out.println("Processed file #" + (i) + " out of " + files.size() + " percentage: " + (i * 100 / files.size()) + "%");
                    // + " output file: " + outputFile);
                }
            } catch (TimeoutException e) {
                System.err.println("Timeout in processing file " + files.get(i));
            }
            ++i;
            if (i * 100 / files.size() >= 99) {
                break;
            }
        }
    }

    public void run(Namespace ns) throws Exception {
        var inputFileName = ns.getString("input");
        var inputDirName = ns.getString("dir");

        PrintStream out = System.out;
        if (ns.getString("output") != null) {
            out = new PrintStream(ns.getString("output"));
        }
        
        try {
            if (inputFileName != null) {
                if (inputFileName.endsWith(".java")) {
                    fileFeaturizer(inputFileName, ns, out);
                } else {
                    lineWiseFeaturizer(inputFileName, ns, out);
                }
            } else if (inputDirName != null) {
                dirFeaturizer(inputDirName, ns, out);
            } else {
                throw new Exception("Either -i or -d must be specified");
            }
        } catch (Exception e) {
            System.err.println("Error1: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
}

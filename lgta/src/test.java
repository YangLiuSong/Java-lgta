import Jama.Matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Map;


public class test {

    public static void main(String[] args) {
        lgta lgtas = new lgta(0.9,3,30);
//        lgta lgtas = new lgta(0.9,3,5);
        lgtas.load_data("src/flickr/landscape_test_data.csv","src/flickr/E.csv","src/flickr/D.csv");
//        lgtas.load_data("src/source/test_data.csv","src/source/E.csv","src/source/D.csv");
        lgtas.train(5);

        lgtas.print_document_topic("document-topic");
        lgtas.TopicWords("topic",10);
    }
}

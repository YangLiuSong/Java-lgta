import java.io.*;
import java.util.*;

import Jama.Matrix;


public class lgta {
    // 声明所有的参数
    // 词语words的集合
    private HashSet<String> words = new HashSet<String>();
    // 单词w在文档d中出现的频数，c(w,d)
    private List<Map<String,Double>> c_w_d = new ArrayList<>();
    // 文档总数
    private int doc_num;
    // 主题数
    private int topic_num;
    // 区域数
    private int region_num;
    // 参数 LambdaB
    private double LambdaB;
    // 总词数
    private double all_count;
    // 各文档经纬度信息 || Shape:doc_num * Matrix
    private List<Matrix> l_d = new ArrayList<>();
    // First-EM 隐藏变量 p(r|d,Variable) || Shape:doc_num * region_num
    private double[][] p_r_dVariable;
    // Second-EM 隐藏变量 sigma(w,r,z) || Shape: topic_num * region_num * words.size()
    private Map<String,Double>[][] sigma_wrz;
    // 地理主题结果p(z|l,Variable) || Shape:doc_num * topic_num
    private double[][] p_z_lVariable;

    public HashSet<String> getWords() {
        return words;
    }

    public List<Map<String, Double>> getC_w_d() {
        return c_w_d;
    }

    public int getDoc_num() {
        return doc_num;
    }

    public double getAll_count() {
        return all_count;
    }

    public List<Matrix> getL_d() {
        return l_d;
    }

    public double[][] getP_r_dVariable() {
        return p_r_dVariable;
    }

    public Map<String, Double>[][] getSigma_wrz() {
        return sigma_wrz;
    }

    public double[][] getP_z_lVariable() {
        return p_z_lVariable;
    }

    public double[][] getP_z_r() {
        return p_z_r;
    }

    public List<Map<String, Double>> getP_w_z() {
        return p_w_z;
    }

    public double[] getP_r_alpha() {
        return p_r_alpha;
    }

    public List<Matrix> getR_E() {
        return r_E;
    }

    public List<Matrix> getR_D() {
        return r_D;
    }

    // EM算法更新的参数
    // 各区域内每个主题的概率分布p(z|r) || List形状：region_num * topic_num
    private double[][] p_z_r;
    // 各主题内每个单词的概率分布p(w|z) || List形状：topic_num * words.size()
    private List<Map<String,Double>> p_w_z = new ArrayList<>();
    // 区域权重概率分布p(r|alpha) || List形状：region_num
    private double[] p_r_alpha;
    // 区域r的期望E与协方差D
    private List<Matrix> r_E = new ArrayList<>();       // Shape:region * Matrix
    private List<Matrix> r_D = new ArrayList<>();       // Shape:region * Matrix
    // 声明所有的参数 End
    private String inputPath;
    // 构造函数
    public lgta(double LambdaB,int topic_num,int region_num){
        this.LambdaB = LambdaB;
        this.topic_num = topic_num;
        this.region_num = region_num;
    }

    // 读取数据
    public boolean load_data(String document_path,String rE_path,String rD_path){
        this.inputPath = document_path;
        double[] r_init = new double[this.region_num];
        double r_sum = 0;
        try {
            BufferedReader csvreader = new BufferedReader(new FileReader(document_path));
            String line = null;
            while((line=csvreader.readLine())!=null){
                String[] items = line.split(",");//CSV格式文件为逗号分隔符文件，这里根据逗号切分
                double[][] l = {{Double.parseDouble(items[0].substring(2)),Double.parseDouble(items[1].substring(0,items[1].length()-2))}};
                Matrix m = new Matrix(l);
                this.l_d.add(m);
                int r_index = Integer.parseInt(items[2]);
                r_init[r_index]++;
                r_sum++;
                String[] ws = items[3].split(" ");
                Map<String,Double> map = new HashMap<>();
                for (String w : ws){
                    this.words.add(w);
                    map.put(w, (map.get(w)==null?1:map.get(w)+1));//核心代码就这一段
                }
                this.c_w_d.add(map);
            }
            this.doc_num = this.c_w_d.size();

            // 计算初始的p(r|alpha)
            this.p_r_alpha = new double[this.region_num];
            for (int r = 0;r < this.region_num;r++){
                this.p_r_alpha[r] = r_init[r] / r_sum;
            }
            // 计算初始的p(r|alpha) End
            // 读取区域的初始分布的期望与方差信息
            BufferedReader E_reader = new BufferedReader(new FileReader(rE_path));
            String E_line = null;
            while((E_line=E_reader.readLine())!=null) {
                String[] items = E_line.split(",");
                double[][] e = {{Double.parseDouble(items[0]),Double.parseDouble(items[1])}};
                Matrix m = new Matrix(e);
                this.r_E.add(m);
            }

            BufferedReader D_reader = new BufferedReader(new FileReader(rD_path));
            String D_line = null;
            while((D_line=D_reader.readLine())!=null){
                String[] items = D_line.split(",");
                String[] i1 = items[0].split(" ");
                String[] i2 = items[1].split(" ");
                double[][] d = {{Double.parseDouble(i1[0].substring(1)),Double.parseDouble(i1[1].substring(0,i1[1].length()-1))},{Double.parseDouble(i2[0].substring(1)),Double.parseDouble(i2[1].substring(0,i2[1].length()-1))}};
                Matrix m = new Matrix(d);
                this.r_D.add(m);
            }
//            double[][] Di = {{1,0},{0,1}};
//            Matrix d_i = new Matrix(Di);
//            for (int rr = 0;rr < this.region_num;rr++){
//                this.r_D.add(d_i);
//            }
            init_model();
            return true;
        }
        catch (Exception e){
            e.printStackTrace();
            return false;
        }
    }

    private void init_model(){
        // 初始化p(z|r)
        this.p_z_r = new double[this.region_num][this.topic_num];
        for (int r = 0;r < this.region_num;r++){
            double[] zr = new double[this.topic_num];
            double zr_sum = 0;
            for (int t = 0;t < this.topic_num;t++){
                zr[t] = Math.random();
                zr_sum += zr[t];
            }
            for (int z = 0;z < this.topic_num;z++){
                this.p_z_r[r][z] = zr[z] / zr_sum;
            }
        }

        // 计算总词数
        this.all_count = 0;
        for (int le = 0;le < this.c_w_d.size();le++){
            this.all_count += this.c_w_d.get(le).size();
        }

        // 给文档中每个词一个初始化的概率分布,初始化p(w|z)
        for (int t = 0;t < this.topic_num;t++){
            Map<String,Double> map = new HashMap<>();
            for (String w : this.words) {
                map.put(w,Math.random());
            }
            this.p_w_z.add(map);
        }
        this.sigma_wrz = new HashMap[this.topic_num][this.region_num];
    }

    public void train(int max_iter){
        // 迭代开始
        long startTime = System.currentTimeMillis();
        for (int epoch = 0;epoch < max_iter;epoch++){
            System.out.println("epoch:" + epoch);
            this.p_r_dVariable = new double[this.doc_num][this.region_num];
            System.out.println("E Step begins starting!");
            // 遍历文档
            for (int d = 0;d < this.doc_num;d++){
                List<Double> numerator = new ArrayList<>();
                for (int r = 0;r < this.region_num;r++){
                    // 计算p(wd|r,Variable)
                    double p_wd_rVariable = 1;
                    // 遍历Map中的Key值，即遍历各文档的单词w
                    Set<Map.Entry<String,Double>> entries = this.c_w_d.get(d).entrySet();
                    for (Map.Entry<String,Double> entry : entries){
                        // 计算p(w|B)
                        double sum_cwd = 0;
                        String w = entry.getKey();
                        for (int d_id = 0;d_id < this.doc_num;d_id++){
                            if (this.c_w_d.get(d_id).containsKey(w)){
                                sum_cwd += this.c_w_d.get(d_id).get(w);
                            }
                        }
                        double p_w_B = sum_cwd / this.all_count;
                        // 计算p(w|B) End

                        // 计算（sum z in Z）p(w|z)p(z|r)
                        double sum_pwz_pzr = 0;
                        for (int t = 0;t < this.topic_num;t++){
                            sum_pwz_pzr += (this.p_z_r[r][t] * this.p_w_z.get(t).get(w));
                        }
                        // 计算（sum z in Z）p(w|z)p(z|r) End

                        // 计算p(w|r,Variable)
                        double p_w_rVariable = this.LambdaB * p_w_B + (1 - this.LambdaB) * sum_pwz_pzr;
                        // 计算p(w|r,Variable) End
                        p_wd_rVariable = p_wd_rVariable * Math.pow(p_w_rVariable, this.c_w_d.get(d).get(w));  // 连乘计算 p(w|r,Varible)的c(w,d)次方
                    }
                    // 计算p(wd|r,Variable) End

                    // 计算p(ld|r,Variable)
                    Matrix m = this.l_d.get(d).minus(this.r_E.get(r));
//                    System.out.println(m.get(0,0) + "   " + m.get(0,1));
                    Matrix m1 = m.times(-1);
                    Matrix m2 = m1.times(this.r_D.get(r).inverse());
                    // m3的行列式过大或过小
                    // Math.exp无法计算
                    Matrix m3 = m2.times(m.transpose());
//                    System.out.println(m3.det());
//                    System.out.println(Math.exp(m3.det()));
                    double p_ld_rVariable = Math.exp(m3.det() / 2)  / (2 * Math.PI * Math.sqrt(Math.abs(this.r_D.get(r).det())));
                    // 计算p(ld|r,Variable) End
                    // 计算 p(wd,ld|r,Variable) = p(ld|r,Variable) * p(wd|r,Variable)
                    double p_wdld_rVariable = p_wd_rVariable * p_ld_rVariable;
                    // 计算 p(wd,ld|r,Variable) = p(ld|r,Variable) * p(wd|r,Variable) End

                    // 计算分子p(r|alpha) * p(wd,ld|r,Variable)
                    numerator.add(this.p_r_alpha[r] * p_wdld_rVariable);
                }
                // 计算分子 End

                // 计算分母
                double denominator = 0;
                for (int i = 0;i<numerator.size();i++){
                    denominator += numerator.get(i);
                }
                // 计算分母 End
                // 计算p(r|d,Variable)
                for (int rr = 0;rr < this.region_num;rr++){
                    this.p_r_dVariable[d][rr] = numerator.get(rr) / denominator;
                }
            }
            // 遍历文档,计算p(r|d,Variable) End
            System.out.println("E Step End!");

            System.out.println("M Step begins starting!");
            for (int r_id = 0;r_id < this.region_num;r_id++){
                double r_alpha_numerator = 0;
                for (int prd_id = 0;prd_id < this.p_r_dVariable.length;prd_id++){
                    r_alpha_numerator += this.p_r_dVariable[prd_id][r_id];
                }
                this.p_r_alpha[r_id] = r_alpha_numerator / this.doc_num;
                double[][] Ezero = {{0,0}};
                double[][] Dzero = {{0,0},{0,0}};
                double[][] j = {{1e-4,0},{0,1e-4}};
                Matrix jm = new Matrix(j);
                Matrix E_numerator = new Matrix(Ezero);
                double denominator = 0;
                Matrix D_numerator = new Matrix(Dzero);
                for (int d_id = 0;d_id < this.doc_num;d_id++){
                    E_numerator = E_numerator.plus(this.l_d.get(d_id).times(this.p_r_dVariable[d_id][r_id]));
                    Matrix m = this.l_d.get(d_id).minus(this.r_E.get(r_id));
                    Matrix m1 = m.transpose().times(m);
                    Matrix m2 = m1.times(this.p_r_dVariable[d_id][r_id]);
                    D_numerator = D_numerator.plus(m2);
                    denominator = denominator + this.p_r_dVariable[d_id][r_id];
                }
                D_numerator = D_numerator.plus(jm);
                System.out.println("E and D");
                System.out.println(E_numerator.times(1/denominator).get(0,0) + "     " + E_numerator.times(1/denominator).get(0,1));
                System.out.println(D_numerator.times(1/denominator).get(0,0) + "     " + D_numerator.times(1/denominator).get(0,1));
                System.out.println(D_numerator.times(1/denominator).get(1,0) + "     " + D_numerator.times(1/denominator).get(1,1));
                this.r_E.set(r_id,E_numerator.times(1/denominator));
                this.r_D.set(r_id,D_numerator.times(1/denominator));
            }
            System.out.println("M Step End!");

            System.out.println("Second EM algorithm is starting!");
            for (int se_epoch = 0;se_epoch < max_iter;se_epoch++){
                // E Step
                // 计算sigma(w,r,z),求解p(z|r)和p(w|z)
                for (int se_z = 0;se_z < this.topic_num;se_z++){
                    for (int se_r = 0;se_r < this.region_num;se_r++){
                        Map<String,Double> map = new HashMap<>();
                        for (String se_w:this.words) {
                            double se_numerator = (1 - this.LambdaB) * this.p_w_z.get(se_z).get(se_w) * this.p_z_r[se_r][se_z];
                            // 计算p(w|B)
                            double sum_cwd = 0;
                            for (int d = 0;d < this.doc_num;d++){
                                if (this.c_w_d.get(d).containsKey(se_w)) {
                                    sum_cwd += this.c_w_d.get(d).get(se_w);
                                }
                            }
                            double se_p_w_B = sum_cwd / this.all_count;
                            // 计算p(w|B) End

                            // 计算（sum z in Z）p(w|z)p(z|r)
                            double se_sum_pwz_pzr = 0;
                            for (int r = 0;r < this.region_num;r++){
                                se_sum_pwz_pzr += (this.p_z_r[r][se_z] * this.p_w_z.get(se_z).get(se_w));
                            }
                            // 计算（sum z in Z）p(w|z)p(z|r) End
                            double se_denominator = this.LambdaB * se_p_w_B + (1 - this.LambdaB) * se_sum_pwz_pzr;
                            map.put(se_w,se_numerator / se_denominator) ;
                        }
                        this.sigma_wrz[se_z][se_r] = map;
                    }
                }
                // E Step End

                // M Step
                // 计算p(z|r)
                for (int r = 0;r < this.region_num;r++){
                    double[] pzr_numerator = new double[this.topic_num];
                    double pzr_denominator = 0;
                    for (int z = 0;z < this.topic_num;z++){
                        double _numerator = 0;
                        for (String _w:this.words) {
                            for (int d = 0;d < this.doc_num;d++){
                                if (this.c_w_d.get(d).containsKey(_w)){
                                    _numerator += (this.c_w_d.get(d).get(_w) * this.p_r_dVariable[d][r] * this.sigma_wrz[z][r].get(_w));
                                }
                            }
                        }
                        pzr_numerator[z] = _numerator;
                        pzr_denominator += _numerator;
                    }
                    for (int zz = 0;zz < this.topic_num;zz++){
                        this.p_z_r[r][zz] = pzr_numerator[zz] / pzr_denominator;
                    }
                }
                // 计算p(z|r) End

                // 计算p(w|z)
                for (int z = 0;z < this.topic_num;z++){
                    Map<String,Double> pwz_numerator = new HashMap<>();
                    double pwz_denominator = 0;
                    for (String w:this.words){
                        double _numerator = 0;
                        for (int d = 0;d < this.doc_num;d++){
                            for (int r = 0;r < this.region_num;r++){
                                if (this.c_w_d.get(d).containsKey(w)){
                                    _numerator += this.c_w_d.get(d).get(w) * this.p_r_dVariable[d][r] * this.sigma_wrz[z][r].get(w);
                                }
                            }
                        }
                        pwz_numerator.put(w,_numerator);
                        pwz_denominator += _numerator;
                    }
                    for (String _w:this.words){
                        this.p_w_z.get(z).put(_w,pwz_numerator.get(_w) / pwz_denominator);
                    }
                }
                // 计算p(w|z) End
                // M Step End
            }
            System.out.println("Second EM algorithm End!");
            long endTime = System.currentTimeMillis();
            System.out.println("iter " + epoch + " costs time: " + (endTime - startTime) + "ms");
        }
        calculate_result();
    }

    // 保存计算的结果
    public void calculate_result(){
        this.p_z_lVariable = new double[this.doc_num][this.topic_num];
        // 计算p(z|l,Variable)
        for (int d = 0;d < this.doc_num;d++){
            for (int t = 0;t < this.topic_num;t++){
                double pz = 0;
                for (int r = 0;r < this.region_num;r++){
                    // 计算p(l|r,Variable)
                    Matrix m = this.l_d.get(d).minus(this.r_E.get(r));
                    Matrix m1 = m.times(-1);
                    Matrix m2 = m1.times(this.r_D.get(r).inverse());
                    // m3的行列式过大或过小
                    // Math.exp无法计算
                    Matrix m3 = m2.times(m.transpose());
                    double p_l_rVariable = Math.exp(m3.det() / 2) / (2 * Math.PI * Math.sqrt(Math.abs(this.r_D.get(r).det())));
                    // 计算p(l|r,Variable) End
                    pz += p_l_rVariable * this.p_z_r[r][t] * this.p_r_alpha[r];
                }
                this.p_z_lVariable[d][t] = pz;
            }
        }
    }

    // 输出每个文档对应的主题信息
    public void print_document_topic(String outputPath){
        int[] pzd = new int[this.doc_num];
        for (int d = 0;d < this.doc_num;d++){
            int maxIndex = 0;
            double max = 0;
            for (int i = 0;i < this.p_z_lVariable[d].length;i++){
                System.out.println(p_z_lVariable[d][i]);
                if (this.p_z_lVariable[d][i] > max) {
                    max = this.p_z_lVariable[d][i];
                    maxIndex = i;
                }
            }
            System.out.println("Document " + d + " Topic Id is :" + maxIndex);
            pzd[d] = maxIndex;
        }
        try {
            BufferedReader csvreader = new BufferedReader(new FileReader(this.inputPath));
            String line = null;
            int index = 0;
            while((line=csvreader.readLine())!=null){
                String[] items = line.split(",");//CSV格式文件为逗号分隔符文件，这里根据逗号切分
                String lat = items[0].substring(2);
                String lon = items[1].substring(0,items[1].length()-2);
                String ws = items[3];
                int topic = pzd[index];
                String writeItem = lat + "," + lon + "," + ws + "," + topic;
                BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputPath + ".txt",true)));
                out.write(writeItem + "\r\n"); // \r\n即为换行
                out.close();
                index++;
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    // 输出主题的前几关键词
    public void TopicWords(String outputPath,int num){
        for (int t = 0;t < this.topic_num;t++){
            String writeItem = "";
            try {
                // Map 按Value进行排序（降序）
                List<Map.Entry<String, Double>> list = new ArrayList<Map.Entry<String, Double>>(this.p_w_z.get(t).entrySet());
                list.sort(new Comparator<Map.Entry<String, Double>>() {
                    @Override
                    public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                        return o2.getValue().compareTo(o1.getValue());
                    }
                });
                System.out.println("Topic " + t + ":");
                writeItem += "Topic " + t + ": \r\n";
                for (int i = 0; i < num; i++) {
                    System.out.println(list.get(i).getKey() + ": " + list.get(i).getValue());
                    writeItem = writeItem + list.get(i).getKey().toString() + ": " + list.get(i).getValue().toString() + "\r\n";
                }
                BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputPath + ".txt",true)));
                out.write(writeItem + "\r\n"); // \r\n即为换行
                out.close();
            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }
}

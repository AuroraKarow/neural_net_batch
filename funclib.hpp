vect vec_travel(vect &vec_val, double (*func)(double&))
{
    auto copy_vec = vec_val;
    for(auto i=0; i<vec_val.ELEM_CNT; ++i) copy_vec.pos_idx(i) = func(vec_val.pos_idx(i));
    return copy_vec;
}

double sigmoid(double &val){ return 1 / (1 + 1 / exp(val)); }

vect sigmoid(vect &vec_val){return vec_travel(vec_val, sigmoid);}

double sigmoid_dv(double &val){ return sigmoid(val) * (1.0 - sigmoid(val)); }

vect sigmoid_dv(vect &vec_val){return vec_travel(vec_val, sigmoid_dv);}

double ReLU(double &val)
{
    if(val < 0) return 0;
    else return val;
}

vect ReLU(vect &vec_val){return vec_travel(vec_val, ReLU);}

double ReLU_dv(double &val)
{
    if(val < 0) return 0;
    else return 1;
}

vect ReLU_dv(vect &vec_val){return vec_travel(vec_val, ReLU_dv);}

vect softmax(vect &vec_val)
{
    vect ans(vec_val.get_ln_cnt(), vec_val.get_col_cnt());
    double sum = 0;
    for(auto i=0; i<vec_val.get_ln_cnt(); ++i)
        for(auto j=0; j<vec_val.get_col_cnt(); ++j)
            sum += std::exp(vec_val[i][j]);
    for(auto i=0; i<vec_val.get_ln_cnt(); ++i)
        for(auto j=0; j<vec_val.get_col_cnt(); ++j)
            ans[i][j] = std::exp(vec_val[i][j]) / sum;
    return ans;
}

vect softmax_dv(vect &vec_input, vect &vec_output)
{
    auto elem_cnt = vec_input.ELEM_CNT;
    if(vec_input.shape_valid(vec_output) && vec_input.LN_CNT==elem_cnt)
    {
        vect ans(elem_cnt, 1);
        for(auto i=0; i<elem_cnt; ++i) for(auto j=0; j<elem_cnt; ++j)
            if(i==j) ans.pos_idx(j) += vec_output.pos_idx(i) * (1 - vec_output.pos_idx(j));
            else ans.pos_idx(j) += (-1) * vec_output.pos_idx(j) * vec_output.pos_idx(i);
        return ans;
    }
    else return blank_vect;
}

vect cec_grad(vect &output, vect &origin)
{
    auto elem_cnt = output.ELEM_CNT;
    vect ans(elem_cnt, 1);
    if(output.shape_valid(origin) && elem_cnt==origin.LN_CNT)
    {
        auto orgn_sum = origin.elem_sum();
        for(auto i=0; i<elem_cnt; ++i)
            ans.pos_idx(i) = (-1) * orgn_sum / output.pos_idx(i);
    } 
    return ans;
}

vect softmax_cec_grad(vect &softmax_output, vect &origin) {return softmax_output - origin;}

vect divisor_dominate(vect &divisor, double epsilon)
{
    auto cpy_val = divisor;
    for(auto i=0; i<cpy_val.get_ln_cnt(); ++i)
        for(auto j=0; j<cpy_val.get_col_cnt(); ++j)
            if(cpy_val[i][j] == 0) cpy_val[i][j] = epsilon;
    return cpy_val;
}

uint64_t samp_block_cnt(uint64_t filter_dir_cnt, uint64_t dir_dilation) {return (dir_dilation + 1) * filter_dir_cnt - dir_dilation;}

uint64_t samp_trace_pos(uint64_t output_dir_pos, uint64_t filter_dir_pos, uint64_t dir_stride, uint64_t dir_dilation) {return output_dir_pos * dir_stride + filter_dir_pos * (1 + dir_dilation);}

uint64_t samp_output_dir_cnt(uint64_t input_dir_cnt, uint64_t filter_dir_cnt, uint64_t dir_stride, uint64_t dir_dilation) {return (input_dir_cnt - samp_block_cnt(filter_dir_cnt, dir_dilation)) / dir_stride + 1;}

uint64_t samp_input_dir_cnt(uint64_t output_dir_cnt, uint64_t filter_dir_cnt, uint64_t dir_stride, uint64_t dir_dilation) {return (output_dir_cnt - 1) * dir_stride + samp_block_cnt(filter_dir_cnt, dir_dilation);}

bool samp_valid(uint64_t input_dir_cnt, uint64_t filter_dir_cnt, uint64_t dir_stride, uint64_t dir_dilation) {return (input_dir_cnt - samp_block_cnt(filter_dir_cnt, dir_dilation)) % dir_stride == 0;}

feature merge_channel(tensor &input)
{
    feature ft_map(input.size());
    for(auto i=0; i<input.size(); ++i) for(auto j=0; j<input[i].size(); ++j)
    {
        if(ft_map[i].is_matrix()) ft_map[i] += input[i][j];
        else ft_map[i] = input[i][j];
        if(!ft_map[i].is_matrix()) return blank_feature;
    }
    return ft_map;
}

set<feature> merge_channel(set<tensor> &input)
{
    set<feature> set_ft_map(input.size());
    for(auto i=0; i<input.size(); ++i) for(auto j=0; j<input[i].size(); ++j)
    {
        set_ft_map[i] = merge_channel(input[i]);
        if(!set_ft_map[i].size()) return blank_ft_seq;
    }
    return set_ft_map;
}

bool deduce_acc_prec_rc(set<vect> &deduce_output, set<uint64_t> &curr_lbl_batch, double net_acc, double &acc, double &prec, double &rc, bool convert_to_rate = true)
{
    if(deduce_output.size() == curr_lbl_batch.size())
    {
        auto confid = 1 - net_acc;
        for(auto i=0; i<curr_lbl_batch.size(); ++i)
        {
            auto curr_prop = deduce_output[i].pos_idx(curr_lbl_batch[i]);
            prec += curr_prop;
            if(curr_prop > 0.5)
            {
                acc += 1;
                if(curr_prop > confid) rc += 1;
            }
        }
        if(convert_to_rate)
        {
            acc /= curr_lbl_batch.size();
            prec /= curr_lbl_batch.size();
            rc /= curr_lbl_batch.size();
        }
        return true;
    }
    else return false;
}

void print_train_status(vect &vecOutput, vect &vecOrgn)
{
    std::cout << " [No.]\t[Output]\t[Origin]" << std::endl;
    for(auto i=0; i<vecOrgn.ELEM_CNT; ++i)
    {
        if(vecOrgn.pos_idx(i)) std::cout << '>';
        else std::cout << ' ';
        std::cout << i << '\t' << vecOutput.pos_idx(i) << '\t' << vecOrgn.pos_idx(i) << std::endl;
    }
}

void print_train_status(set<vect> &setOutput, set<vect> &setOrgn)
{
    for(auto i=0; i<setOutput.size(); ++i)
    {
        print_train_status(setOutput[i], setOrgn[i]);
        std::cout << std::endl;
    }
}

void print_train_status(int epoch, int curr_prog, int prog, double acc, double prec, double rc, int dur) { std::printf("\r[Ep][%d][Prog][%d/%d][Acc/Prec/Rc][%.2f/%.2f/%.2f][Dur][%dms]", epoch, curr_prog, prog, acc, prec, rc, dur); }

void print_deduce_status(int epoch, double acc, double prec, double rc, int dur)
{ std::printf("\r[Ep][%d][Acc/Prec/Rc][%lf/%lf/%lf][Dur][%dms]", epoch, acc, prec, rc, dur); }

void print_deduce_progress(int curr_prog, int prog) { std::printf("\r[Deducing][%d/%d]", curr_prog, prog); }
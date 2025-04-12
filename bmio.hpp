BMIO_BEGIN

class bitmap final
{
private:
    BMIO_CHANN R;
    BMIO_CHANN G;
    BMIO_CHANN B;
    BMIO_CHANN A;
    GDI_STARTUP_INPUT st_gph;
    GDI_TOKEN gph_token;
    BMIO_STR dir_path(BMIO_STR dir_path_root, BMIO_STR name, char div_syb = '\\')
    {
        if(div_syb == '\\') return dir_path_root + '\\' + name + ".bmp";
        else if(div_syb == '/') return dir_path_root + '/' + name + ".bmp";
        else return "";
    }
    bool init_chann(uint64_t ln_cnt, uint64_t col_cnt)
    {
        if(ln_cnt && col_cnt)
        {
            R = BMIO_CHANN(ln_cnt, col_cnt);
            G = BMIO_CHANN(ln_cnt, col_cnt);
            B = BMIO_CHANN(ln_cnt, col_cnt);
            A = BMIO_CHANN(ln_cnt, col_cnt);
            return true;
        }
        else return false;
    }
    bool set_chann(BMIO_CHANN &R_src, BMIO_CHANN &G_src, BMIO_CHANN &B_src, BMIO_CHANN &A_src, bool move_flag = false)
    {
        if(R_src.shape_valid(G_src) && G_src.shape_valid(B_src) && B_src.shape_valid(A_src))
        {
            if(move_flag)
            {
                R = std::move(R_src);
                G = std::move(G_src);
                B = std::move(B_src);
                A = std::move(A_src);
            }
            else
            {
                R = R_src;
                G = G_src;
                B = B_src;
                A = A_src;
            }
            return true;
        }
        else return false;
    }
    bool CLSID_encode(const WCHAR* format, CLSID* p_CLSID)
    {
        // Number of image encoders
        uint32_t  num = 0;
        // Size of the image encoder array in bytes
        uint32_t  size = 0;
        Gdiplus::GetImageEncodersSize(&num, &size);
        if(size)
        {
            auto img_code_info_ptr = new Gdiplus::ImageCodecInfo[size];
            if(img_code_info_ptr)
            {
                Gdiplus::GetImageEncoders(num, size, img_code_info_ptr);
                for(auto i=0; i<num; ++i)
                {
                    if(!wcscmp(img_code_info_ptr[i].MimeType, format))
                    {
                        *p_CLSID = img_code_info_ptr[i].Clsid;
                        delete [] img_code_info_ptr;
                        img_code_info_ptr = nullptr;
                        return true;
                    }
                }
            }
            delete [] img_code_info_ptr;
            img_code_info_ptr = nullptr;
        }
        return false;
    }
    bool refresh_chann(uint64_t _chann)
    {
        switch (_chann)
        {
        case BMIO_R:
            if(R.is_matrix()) R.value_fill(0);
            return true;
        case BMIO_G:
            if(G.is_matrix()) G.value_fill(0);
            return true;
        case BMIO_B:
            if(B.is_matrix()) B.value_fill(0);
            return true;
        case BMIO_A:
            if(A.is_matrix()) A.value_fill(0);
            return true;
        default: return false;
        }
    }
    bool refresh_chann() {return (refresh_chann(BMIO_R) && refresh_chann(BMIO_G) && refresh_chann(BMIO_B) && refresh_chann(BMIO_A));}
public:
    bitmap() {}
    bool set_size(uint64_t ln_cnt, uint64_t col_cnt) { return init_chann(ln_cnt, col_cnt); }
    bool set_raw(BMIO_CHANN &R_src, BMIO_CHANN &G_src, BMIO_CHANN &B_src, BMIO_CHANN &A_src, bool move_flag = false) { return set_chann(R_src, G_src, B_src, A_src, move_flag); }
    void reset()
    {
        R.reset();
        G.reset();
        B.reset();
        A.reset();
    }
    uint64_t ln_cnt() { return R.LN_CNT; }
    uint64_t col_cnt() { return R.COL_CNT; }
    __declspec (property (get=ln_cnt)) uint64_t HEIGHT;
    __declspec (property (get=col_cnt)) uint64_t WIDTH;
    bool img_valid() { return HEIGHT && WIDTH && R.shape_valid(G) && G.shape_valid(B); }
    bitmap(bitmap &val) { *this = val; }
    bitmap(bitmap &&val) { *this = std::move(val); }
    bitmap(BMIO_RAW &&vec) { if(vec.size() == BMIO_RGB_CNT) set_chann(vec[BMIO_R], vec[BMIO_G], vec[BMIO_B], vec[BMIO_A], true); }
    bitmap(BMIO_RAW &vec) { if(vec.size() == BMIO_RGB_CNT) set_chann(vec[BMIO_R], vec[BMIO_G], vec[BMIO_B], vec[BMIO_A]); }
    bool load_img(BMIO_WSTR dir, bool rgba = false)
    {
        GDI_STARTUP(&gph_token, &st_gph, nullptr);
        GDI_BITMAP map_ptr(dir.c_str());
        init_chann(map_ptr.GetHeight(), map_ptr.GetWidth());
        for(uint64_t i=0; i<map_ptr.GetHeight(); ++i)
            for(uint64_t j=0; j<map_ptr.GetWidth(); ++j)
            {
                GDI_COLOR color;
                if(map_ptr.GetPixel(j, i, &color) == GDI_STATUS::Ok)
                {
                    R[i][j] = color.GetRed();
                    G[i][j] = color.GetGreen();
                    B[i][j] = color.GetBlue();
                    if(rgba) A[i][j] = color.GetAlpha();
                }
                else
                {
                    refresh_chann();
                    return false;
                }
            }
        return true;
    }
    bool load_img(BMIO_STR dir, bool rgba = false) { return load_img(BMIO_CHARSET(dir), rgba); }
    bitmap(BMIO_STR dir, bool rgba = false) { load_img(dir, rgba); }
    bitmap(BMIO_WSTR dir, bool rgba = false) { load_img(dir, rgba); }
    bool save_img(BMIO_WSTR dir_root, BMIO_WSTR name, uint64_t extend, bool rgba = false, wchar_t div_syb = L'\\')
    {
        if(img_valid())
        {
            GDI_STARTUP(&gph_token, &st_gph, nullptr);
            auto color_form = PixelFormat32bppRGB;
            if(rgba) color_form = PixelFormat32bppARGB;
            GDI_BITMAP bitmap(WIDTH, HEIGHT, color_form);
            GDI_GRAPHICS gph_img(&bitmap);
            CLSID CID_STR;
            BMIO_WSTR ext_name = L"";
            switch (extend)
            {
            case BMIO_PNG:
                if(CLSID_encode(L"image/png", &CID_STR))
                {
                    ext_name = L".png";
                    break;
                }
                else return false;
            case BMIO_JPG:
                if(CLSID_encode(L"image/jpeg", &CID_STR) && !rgba)
                {
                    ext_name = L".jpg";
                    break;
                }
                else return false;
            case BMIO_GIF:
                if(CLSID_encode(L"image/gif", &CID_STR))
                {
                   ext_name = L".gif";
                   break; 
                }
                else return false;
            case BMIO_TIF:
                if(CLSID_encode(L"image/tiff", &CID_STR))
                {
                    ext_name = L".tiff";
                    break;
                }
                else return false;
            case BMIO_BMP:
                if(CLSID_encode(L"image/bmp", &CID_STR) && !rgba)
                {
                    ext_name = L".bmp";
                    break;
                }
                else return false;
            default: return false;
            }
            // Draw image
            for(auto i=0; i<HEIGHT; ++i)
                for(auto j=0; j<WIDTH; ++j)
                    if(rgba) gph_img.DrawLine(&GDI_PEN(GDI_COLOR(A[i][j], R[i][j], G[i][j], B[i][j])), j, i, j+1, i+1);
                    else gph_img.DrawLine(&GDI_PEN(GDI_COLOR(R[i][j], G[i][j], B[i][j])), j, i, j+1, i+1);
            BMIO_WSTR path = dir_root, file_name = name + ext_name;
            if(div_syb == L'\\') path += L'\\' + file_name;
            else path += L'/' + file_name;
            return GDI_STATUS::Ok == bitmap.Save(path.c_str(), &CID_STR);
        }
        else return false;
    }
    bool save_img(BMIO_STR dir_root, BMIO_STR name, uint64_t extend, bool rgba = false, char div_syb = '\\') { return save_img(BMIO_CHARSET(dir_root), BMIO_CHARSET(name), extend, rgba, div_syb); }
    BMIO_CHANN gray()
    {
        if(img_valid()) return BMIO_GRAY_WEIGHT_R * R + BMIO_GRAY_WEIGHT_G * G + BMIO_GRAY_WEIGHT_B * B;
        else return BMIO_CHANN::blank_matrix();
    }
    static BMIO_RAW gray_grad(BMIO_CHANN &grad_vec)
    {
        if(grad_vec.is_matrix())
        {
            BMIO_RAW RGB_gradient(BMIO_RGB_CNT);
            RGB_gradient[BMIO_R] = grad_vec * BMIO_GRAY_WEIGHT_R;
            RGB_gradient[BMIO_G] = grad_vec * BMIO_GRAY_WEIGHT_G;
            RGB_gradient[BMIO_B] = grad_vec * BMIO_GRAY_WEIGHT_B;
            return RGB_gradient;
        }
        else return BMIO_RAW::blank_queue();
    }
    BMIO_RAW img_vec(bool rgba = false)
    {
        auto chann_cnt = BMIO_RGB_CNT;
        if(rgba) chann_cnt = BMIO_RGBA_CNT;
        BMIO_RAW raw_vec(chann_cnt);
        raw_vec[BMIO_R] = R;
        raw_vec[BMIO_G] = G;
        raw_vec[BMIO_B] = B;
        if(rgba) raw_vec[BMIO_A] = A;
        return raw_vec;
    }
    bool operator==(bitmap &val) { return R==val.R && G==val.G && B==val.B && A==val.A; }
    bool operator!=(bitmap &val) { return !(*this == val); }
    void operator=(bitmap &val) { set_chann(val.R, val.G, val.B, val.A); }
    void operator=(bitmap &&val) { set_chann(val.R, val.G, val.B, val.A, true); }
    ~bitmap() { reset(); }
};

BMIO_END
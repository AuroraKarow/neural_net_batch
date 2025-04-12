CONV_BEGIN

vect Conv(vect &vecInput, vect &vecKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    if(SAMP_VALID(vecInput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation) &&
        SAMP_VALID(vecInput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation) &&
        vecInput.is_matrix() && vecKernel.is_matrix())
    {
        auto vecPrepInput = vecInput.pad(iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        auto iOutputLnCnt = SAMP_OUTPUT_DIR_CNT(vecPrepInput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation),
            iOutputColCnt = SAMP_OUTPUT_DIR_CNT(vecPrepInput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation);
        vect vecOutput(iOutputLnCnt, iOutputColCnt);
        for(auto i=0; i<iOutputLnCnt; ++i)
            for(auto j=0; j<iOutputColCnt; ++j)
                for(auto k=0; k<vecKernel.LN_CNT; ++k)
                    for(auto l=0; l<vecKernel.COL_CNT; ++l)
                        vecOutput[i][j] += vecPrepInput[SAMP_TRACE_POS(i, k, iLnStride, iLnDilation)][SAMP_TRACE_POS(j, l, iColStride, iColDilation)] * vecKernel[k][l];
        return vecOutput;
    }
    else return blank_vect;
}

feature Conv(feature &vecInput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    feature vecOutput(tenKernel.size());
    for(auto i=0; i<tenKernel.size(); ++i) for(auto j=0; j<vecInput.size(); ++j) 
    {
        auto vecSglMap = Conv(vecInput[j], tenKernel[i][j], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(vecOutput[i].is_matrix()) vecOutput[i] += vecSglMap;
        else vecOutput[i] = std::move(vecSglMap);
        if(!vecOutput[i].is_matrix()) return blank_feature;
    }
    return vecOutput;
}

vect GradLossToKernel(vect &vecGradLossToOutput, vect &vecInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    auto vecPrepInput = vecInput.pad(iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    return Conv(vecPrepInput, vecGradLossToOutput, iLnStride, iColStride, iLnDilation, iColDilation);
}

tensor GradLossToKernel(feature &vecGradLossToOutput, feature &vecInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    tensor tenGrad(vecGradLossToOutput.size());
    for(auto i=0; i<vecGradLossToOutput.size(); ++i)
    {
        auto iChannCnt = vecInput.size();
        tenGrad[i].init(iChannCnt);
        for(auto j=0; j<iChannCnt; ++j)
        {
            tenGrad[i][j] = GradLossToKernel(vecGradLossToOutput[i], vecInput[j], iLnStride, iColStride, iLnDilation, iColDilation);
            if(!tenGrad[i][j].is_matrix()) return blank_tensor;
        }
    }
    return tenGrad;
}

vect GradLossToInput(vect &vecGradLossToOutput, vect &vecKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    auto iInputLnCnt = SAMP_INPUT_DIR_CNT(vecGradLossToOutput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation),
        iInputColCnt = SAMP_INPUT_DIR_CNT(vecGradLossToOutput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation);
    vect vecGrad(iInputLnCnt, iInputColCnt);
    for(auto i=0; i<vecGradLossToOutput.LN_CNT; ++i)
        for(auto j=0; j<vecGradLossToOutput.COL_CNT; ++j)
            for(auto k=0; k<vecKernel.LN_CNT; ++k)
                for(auto l=0; l<vecKernel.COL_CNT; ++l)
                    vecGrad[SAMP_TRACE_POS(i, k, iLnStride, iLnDilation)][SAMP_TRACE_POS(j, l, iColStride, iColDilation)] += vecGradLossToOutput[i][j] * vecKernel[k][l];
    return vecGrad.crop(iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
}

feature GradLossToInput(feature &vecGradLossToOutput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    feature vecGrad;
    if(vecGradLossToOutput.size() == tenKernel.size()) for(auto i=0; i<tenKernel.size(); ++i)
    {
        if(!vecGrad.size()) vecGrad.init(tenKernel[i].size());
        for(auto j=0; j<tenKernel[i].size(); ++j)
        {
            auto vecSglGrad = GradLossToInput(vecGradLossToOutput[i], tenKernel[i][j], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
            if(vecSglGrad.is_matrix())
                if(vecGrad[j].is_matrix()) vecGrad[j] += vecSglGrad;
                else vecGrad[j] = std::move(vecSglGrad);
            else return blank_feature;
        }
    }
    return vecGrad;
}

vect PoolDownMaxAvg(vect &vecInput, uint64_t iFilterLnCnt, uint64_t iFilterColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iPoolType = POOL_DOWN_MAX, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    vect vecOutput;
    if(SAMP_VALID(vecInput.LN_CNT, iFilterLnCnt, iLnStride, iLnDilation) &&
    SAMP_VALID(vecInput.COL_CNT, iFilterColCnt, iColStride, iColDilation))
    {
        auto iOutputLnCnt = SAMP_OUTPUT_DIR_CNT(vecInput.LN_CNT, iFilterLnCnt, iLnStride, iLnDilation),
            iOutputColCnt = SAMP_OUTPUT_DIR_CNT(vecInput.COL_CNT, iFilterColCnt, iColStride, iColDilation);
        vecOutput = vect(iOutputLnCnt, iOutputColCnt);
        for(auto i=0; i<vecOutput.LN_CNT; ++i)
            for(auto j=0; j<vecOutput.COL_CNT; ++j)
            {
                double dPoolElem = 0;
                if(iPoolType == POOL_DOWN_MAX) dPoolElem = vecInput.extremum(SAMP_TRACE_POS(i, 0, iLnStride, iLnDilation), SAMP_TRACE_POS(i, iFilterLnCnt-1, iLnStride, iLnDilation), SAMP_TRACE_POS(j, 0, iColStride, iColDilation), SAMP_TRACE_POS(j, iFilterColCnt-1, iColStride, iColDilation), iLnDilation, iColDilation).val;
                else if(iPoolType == POOL_DOWN_AVG) dPoolElem = vecInput.elem_sum(SAMP_TRACE_POS(i, 0, iLnStride, iLnDilation), SAMP_TRACE_POS(i, iFilterLnCnt-1, iLnStride, iLnDilation), SAMP_TRACE_POS(j, 0, iColStride, iColDilation), SAMP_TRACE_POS(j, iFilterColCnt-1, iColStride, iColDilation), iLnDilation, iColDilation) / (iFilterLnCnt * iFilterColCnt);
                else return blank_vect;
                vecOutput[i][j] = dPoolElem;
            }
    }
    return vecOutput;
}

vect PoolDownGlbAvg(vect &vecInput) {return vect(vecInput.elem_sum()/vecInput.ELEM_CNT);}

feature PoolDown(feature &vecInput, uint64_t iPoolType = POOL_DOWN_MAX, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    feature vecOutput(vecInput.size());
    for(auto i=0; i<vecInput.size(); ++i)
    {
        if(iPoolType == POOL_DOWN_GAG) vecOutput[i] = PoolDownGlbAvg(vecInput[i]);
        else vecOutput[i] = PoolDownMaxAvg(vecInput[i], iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iPoolType, iLnDilation, iColDilation);
        if(!vecOutput[i].is_matrix()) return blank_feature;
    }
    return vecOutput;
}

vect PoolUpGlbAvg(vect &vecInput, uint64_t iUpLnCnt, uint64_t iUpColCnt)
{
    if(vecInput.ELEM_CNT == 1)
    {
        vect vecOuput(iUpLnCnt, iUpColCnt);
        vecOuput.value_fill(vecInput.atom() / vecOuput.ELEM_CNT);
        return vecOuput;
    }
    else return blank_vect;
}

vect PoolUpMaxAvg(vect &vecInput, uint64_t iFilterLnCnt, uint64_t iFilterColCnt, uint64_t iLnStride, uint64_t iColStride, vect &vecTraceInput = vect(), uint64_t iPoolType = POOL_UP_MAX, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    auto iOutputLnCnt = 0, iOutputColnCnt = 0;
    if(vecTraceInput.is_matrix())
    {
        iOutputLnCnt = vecTraceInput.LN_CNT;
        iOutputColnCnt = vecTraceInput.COL_CNT;
    }
    else
    {
        iOutputLnCnt = SAMP_INPUT_DIR_CNT(vecInput.LN_CNT, iFilterLnCnt, iLnStride, iLnDilation);
        iOutputColnCnt = SAMP_INPUT_DIR_CNT(vecInput.COL_CNT, iFilterColCnt, iColStride, iColDilation);
    }
    vect vecOutput(iOutputLnCnt, iOutputColnCnt);
    for(auto i=0; i<vecInput.LN_CNT; ++i) for(auto j=0; j<vecInput.COL_CNT; ++j)
        if(iPoolType==POOL_UP_AVG)
        {
            auto dElemGrad = vecInput[i][j] / (iFilterLnCnt * iFilterColCnt);
            for(auto k=0; k<iFilterLnCnt; ++k) for(auto l=0; l<iFilterColCnt; ++l) vecOutput[SAMP_TRACE_POS(i, k, iLnStride, iLnDilation)][SAMP_TRACE_POS(j, l, iColStride, iColDilation)] = dElemGrad;
        }
        else if(iPoolType==POOL_UP_MAX && vecTraceInput.is_matrix())
        {
            auto posExtrm = vecTraceInput.extremum(SAMP_TRACE_POS(i, 0, iLnStride, iLnDilation), SAMP_TRACE_POS(i, iFilterLnCnt-1, iLnStride, iLnDilation), SAMP_TRACE_POS(j, 0, iColStride, iColDilation), (j, iFilterColCnt-1, iColStride, iColDilation), iLnDilation, iColDilation);
            for(auto k=0; k<posExtrm.pos_list.size(); ++k) vecOutput[posExtrm.pos_list[k].ln][posExtrm.pos_list[k].col] += vecInput[i][j];
        }
        else return blank_vect;
    return vecOutput;
}

feature PoolUp(feature &vecInput, uint64_t iPoolType = POOL_UP_MAX, feature &vecTraceInput = feature(), uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    feature vecOutput(vecInput.size());
    for(auto i=0; i<vecInput.size(); ++i) 
    {
        if(iPoolType == POOL_UP_GAG) vecOutput[i] = PoolUpGlbAvg(vecInput[i], vecTraceInput[i].LN_CNT, vecTraceInput[i].COL_CNT);
        else vecOutput[i] = PoolUpMaxAvg(vecInput[i], iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, vecTraceInput[i], iPoolType, iLnDilation, iColDilation);
        if(!vecOutput[i].is_matrix()) return blank_feature;
    }
    return vecOutput;
}

vect Im2ColFeaturePad(vect &vecIm2ColSrc, uint64_t &iLnPadCnt, uint64_t &iColPadCnt, uint64_t iLnCnt, uint64_t iColCnt, uint64_t iTopCnt, uint64_t iRightCnt, uint64_t iBottomCnt, uint64_t iLeftCnt, uint64_t iLnDistance, uint64_t iColDistance)
{
    if(iTopCnt || iRightCnt || iBottomCnt || iLeftCnt || iLnDistance || iColDistance)
    {
        if(!iLnPadCnt) iLnPadCnt = mtx::mtx_pad_cnt(iTopCnt, iBottomCnt, iLnCnt, iLnDistance);
        if(!iColPadCnt) iColPadCnt = mtx::mtx_pad_cnt(iLeftCnt, iRightCnt, iColCnt, iColDistance);
        vect vecAns(iLnPadCnt*iColPadCnt, vecIm2ColSrc.COL_CNT);
        for(auto i=0; i<vecIm2ColSrc.ELEM_CNT; ++i)
        {
            auto posCurrAxis = mtx::mtx_elem_pos(i, vecIm2ColSrc.COL_CNT);
            auto posCurrDim = mtx::mtx_elem_pos(posCurrAxis.ln, iColCnt);
            vecAns[mtx::mtx_elem_pos(iTopCnt+posCurrDim.ln*(iLnDistance+1),iLeftCnt+posCurrDim.col*(iColDistance+1),iColPadCnt)][posCurrAxis.col] = vecIm2ColSrc.pos_idx(i);
        }
        return vecAns;
    }
    else { iLnPadCnt = iLnCnt; iColPadCnt = iColCnt; return vecIm2ColSrc; }
}

vect Im2ColFeatureCrop(vect &vecIm2ColSrc, uint64_t &iLnCropCnt, uint64_t &iColCropCnt, uint64_t iLnCnt, uint64_t iColCnt, uint64_t iTopCnt, uint64_t iRightCnt, uint64_t iBottomCnt, uint64_t iLeftCnt, uint64_t iLnDistance, uint64_t iColDistance)
{
    if(iTopCnt || iRightCnt || iBottomCnt || iLeftCnt || iLnDistance || iColDistance)
    {
        if(!iLnCropCnt) iLnCropCnt = mtx::mtx_crop_cnt(iTopCnt, iBottomCnt, iLnCnt, iLnDistance);
        if(!iColCropCnt) iColCropCnt = mtx::mtx_crop_cnt(iLeftCnt, iRightCnt, iColCnt, iColDistance);
        vect vecAns(iLnCropCnt*iColCropCnt, vecIm2ColSrc.COL_CNT);
        for(auto i=0; i<vecAns.ELEM_CNT; ++i)
        {
            auto posCurrAxis = mtx::mtx_elem_pos(i, vecAns.COL_CNT);
            auto posCurrDim = mtx::mtx_elem_pos(posCurrAxis.ln, iColCropCnt);
            vecAns.pos_idx(i) = vecIm2ColSrc[mtx::mtx_elem_pos(iTopCnt+posCurrDim.ln*(iLnDistance+1),iLeftCnt+posCurrDim.col*(iColDistance+1), iColCnt)][posCurrAxis.col];
        }
        return vecAns;
    }
    else { iLnCropCnt = iLnCnt; iColCropCnt = iColCnt; return vecIm2ColSrc; }
}

vect Im2ColInputCaffeTransform(vect &vecIm2ColSrc, uint64_t iInputLnCnt, uint64_t iInputColCnt, uint64_t &iOutputLnCnt, uint64_t &iOutputColCnt, uint64_t iFilterLnCnt, uint64_t iFilterColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation, uint64_t iColDilation)
{
    if(!iOutputLnCnt) iOutputLnCnt = SAMP_OUTPUT_DIR_CNT(iInputLnCnt, iFilterLnCnt, iLnStride, iLnDilation);
    if(!iOutputColCnt) iOutputColCnt = SAMP_OUTPUT_DIR_CNT(iInputColCnt, iFilterColCnt, iColStride, iColDilation);
    auto iOutputElemCnt = iOutputLnCnt * iOutputColCnt,
        iFilterElemCnt = iFilterLnCnt * iFilterColCnt;
    vect vecAns(iOutputElemCnt, vecIm2ColSrc.COL_CNT*iFilterElemCnt);
    auto iTensorSize = iOutputElemCnt * iFilterElemCnt,
        iHyperSize = vecIm2ColSrc.COL_CNT * iTensorSize;
    for(auto i=0; i<iHyperSize; ++i)
    {
        auto posCurrHyperDim = mtx::mtx_elem_pos(i, iTensorSize),
            posCurrTensorDim = mtx::mtx_elem_pos(posCurrHyperDim.col, iFilterElemCnt);
        auto iOutputPos = mtx::mtx_elem_pos(posCurrTensorDim.ln, iOutputColCnt),
            iFilterPos = mtx::mtx_elem_pos(posCurrTensorDim.col, iFilterColCnt);
        vecAns[posCurrTensorDim.ln][posCurrHyperDim.ln*iFilterElemCnt+posCurrTensorDim.col] = vecIm2ColSrc[mtx::mtx_elem_pos(SAMP_TRACE_POS(iOutputPos.ln, iFilterPos.ln, iLnStride, iLnDilation),SAMP_TRACE_POS(iOutputPos.col, iFilterPos.col, iColStride, iColDilation),iInputColCnt)][posCurrHyperDim.ln];
    }
    return vecAns;
}

// [iInputLnCnt][iInputColCnt] could be blank for gradient calculation otherwise should be relaid with [iOutputLnCnt]
vect Im2ColInputCaffeTransform(vect &vecIm2ColCaffeSrc, uint64_t &iInputLnCnt, uint64_t &iInputColCnt, uint64_t iOutputLnCnt, uint64_t iOutputColCnt, uint64_t iFilterLnCnt, uint64_t iFilterColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation, uint64_t iColDilation, bool bGradFlag)
{
    if(!iInputLnCnt) iInputLnCnt = SAMP_INPUT_DIR_CNT(iOutputLnCnt, iFilterLnCnt, iLnStride, iLnDilation);
    if(!iInputColCnt) iInputColCnt = SAMP_INPUT_DIR_CNT(iOutputColCnt, iFilterColCnt, iColStride, iColDilation);
    auto iFilterElemCnt = iFilterLnCnt * iFilterColCnt;
    vect vecIm2ColAns(iInputLnCnt*iInputColCnt, vecIm2ColCaffeSrc.COL_CNT/iFilterElemCnt);
    for(auto i=0; i<vecIm2ColAns.COL_CNT; ++i) for(auto j=0; j<vecIm2ColCaffeSrc.LN_CNT; ++j) for(auto k=0; k<iFilterElemCnt; ++k)
    {
        auto iOutputPos = mtx::mtx_elem_pos(j, iOutputColCnt),
            iFilterPos = mtx::mtx_elem_pos(k, iFilterColCnt);
        auto iTraceIdx = mtx::mtx_elem_pos(SAMP_TRACE_POS(iOutputPos.ln, iFilterPos.ln, iLnStride, iLnDilation), SAMP_TRACE_POS(iOutputPos.col, iFilterPos.col, iColStride, iColDilation), iInputColCnt);
        if(bGradFlag) vecIm2ColAns[iTraceIdx][i] += vecIm2ColCaffeSrc[j][i*iFilterElemCnt+k];
        else vecIm2ColAns[iTraceIdx][i] = vecIm2ColCaffeSrc[j][i*iFilterElemCnt+k];
    }
    return vecIm2ColAns;
}

vect ConvIm2Col(vect &vecIm2ColCaffeInput, vect &vecIm2ColKernel) { return fc::Output(vecIm2ColKernel, vecIm2ColCaffeInput); }

vect GradLossToConvIm2ColCaffeInput(vect &vecGradLossToOutput, vect &vectKernel) { return fc::GradLossToWeight(vecGradLossToOutput, vectKernel); }

vect GradLossToKernelIm2Col(vect &vecGradLossToOutput, vect &vecIm2ColCaffeInput) { return fc::GradLossToInput(vecGradLossToOutput, vecIm2ColCaffeInput); }

vect PoolGlbAvgIm2Col(vect &vecInput)
{
    vect vecAns(IDX_SGL, vecInput.COL_CNT);
    for(auto i=0; i<vecAns.COL_CNT; ++i) vecAns[IDX_ZERO][i] = vecInput.elem_sum(0, vecInput.LN_CNT-1, i, i) / vecInput.LN_CNT;
    return vecAns;
}

vect GradLossToPoolGlbAvgInputIm2Col(vect &vecGradLossToOutput, uint64_t iChannElemCnt)
{
    vect vecGrad(iChannElemCnt, vecGradLossToOutput.COL_CNT);
    vecGradLossToOutput *= ((1.0) / iChannElemCnt);
    for(auto i=0; i<vecGrad.ELEM_CNT; ++i) vecGrad.pos_idx(i) = vecGradLossToOutput.pos_idx(mtx::mtx_elem_pos(i, vecGrad.COL_CNT).col);
    return vecGrad;
}

vect PoolMaxAvgIm2Col(uint64_t iPoolType, vect &vecIm2ColCaffeInput, set<bagrt::net_list<mtx::mtx_pos>> &setIm2ColInputPoolExtmPosList, uint64_t iFilterLnCnt, uint64_t iFilterColCnt)
{
    auto iFilterElemCnt = iFilterLnCnt * iFilterColCnt;
    vect vecAns(vecIm2ColCaffeInput.LN_CNT, vecIm2ColCaffeInput.COL_CNT/iFilterElemCnt);
    for(auto i=0; i<vecAns.ELEM_CNT; ++i)
    {
        auto posCurrDim = mtx::mtx_elem_pos(i, vecAns.COL_CNT);
        auto iCurrChannBeginCol = posCurrDim.col * iFilterElemCnt;
        auto iCurrVal = 0.0;
        if(iPoolType == POOL_AVG_IM2COL) iCurrVal = vecIm2ColCaffeInput.elem_sum(posCurrDim.ln, posCurrDim.ln, iCurrChannBeginCol, iCurrChannBeginCol+iFilterElemCnt-1) / iFilterElemCnt;
        else if(iPoolType == POOL_MAX_IM2COL)
        {
            auto iExtmTemp = vecIm2ColCaffeInput.extremum(posCurrDim.ln, posCurrDim.ln, iCurrChannBeginCol, iCurrChannBeginCol+iFilterElemCnt-1, true, setIm2ColInputPoolExtmPosList.size());
            iCurrVal = iExtmTemp.val;
            if(setIm2ColInputPoolExtmPosList.size()) setIm2ColInputPoolExtmPosList[i] = std::move(iExtmTemp.pos_list);
        }
        else return blank_vect;
        vecAns.pos_idx(i) = iCurrVal;
    }
    return vecAns;
}

vect GradLossToPoolMaxAvgIm2ColCaffeInput(uint64_t iPoolType, vect &vecGradLossToOutput, set<bagrt::net_list<mtx::mtx_pos>> &setIm2ColInputPoolExtmPosList, uint64_t iFilterLnCnt, uint64_t iFilterColCnt)
{
    auto iFilterElemCnt = iFilterLnCnt * iFilterColCnt;
    vect vecCaffeGrad(vecGradLossToOutput.LN_CNT, iFilterElemCnt*vecGradLossToOutput.COL_CNT);
    for(auto i=0; i<vecGradLossToOutput.ELEM_CNT; ++i)
    {
        auto posCurrDim = mtx::mtx_elem_pos(i, vecGradLossToOutput.COL_CNT);
        auto iCurrChannBeginCol = posCurrDim.col * iFilterElemCnt;
        if(iPoolType == POOL_AVG_IM2COL)
        {
            auto iAvgVal = vecGradLossToOutput.pos_idx(i) / iFilterElemCnt;
            for(auto j=0; j<iFilterElemCnt; ++j) vecCaffeGrad[posCurrDim.ln][j+iCurrChannBeginCol] = iAvgVal;
        }
        else if(iPoolType==POOL_MAX_IM2COL && setIm2ColInputPoolExtmPosList.size()) for(auto j=0; j<setIm2ColInputPoolExtmPosList[i].size(); ++j) vecCaffeGrad[setIm2ColInputPoolExtmPosList[i][j].ln][setIm2ColInputPoolExtmPosList[i][j].col] += vecGradLossToOutput.pos_idx(i);
        else return blank_vect;
    }
    return vecCaffeGrad;
}

CONV_END
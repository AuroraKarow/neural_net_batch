FC_BEGIN

vect InitWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
{
    if(iInputLnCnt && iOutputLnCnt) return vect(iOutputLnCnt, iInputLnCnt, true, dRandBoundryFirst, dRandBoundrySecond, dAcc);
    else return blank_vect;
}

struct FCBN
{
    vect vecMuBeta;
    vect vecSigmaSqr;
    set<vect> setBarX;
    set<vect> setY;
    FCBN(){}
    void ValueCopy(FCBN &FCBNVal)
    {
        vecMuBeta = FCBNVal.vecMuBeta;
        vecSigmaSqr = FCBNVal.vecSigmaSqr;
        setBarX = FCBNVal.setBarX;
        setY = FCBNVal.setY;
    }
    void ValueMove(FCBN &&FCBNVal)
    {
        vecMuBeta = std::move(FCBNVal.vecMuBeta);
        vecSigmaSqr = std::move(FCBNVal.vecSigmaSqr);
        setBarX = std::move(FCBNVal.setBarX);
        setY = std::move(FCBNVal.setY);
    }
    FCBN(FCBN &FCBNVal) { ValueCopy(FCBNVal); }
    FCBN(FCBN &&FCBNVal) { ValueMove(std::move(FCBNVal)); }
    void operator=(FCBN &FCBNVal) { ValueCopy(FCBNVal); }
    void operator=(FCBN &&FCBNVal) { ValueMove(std::move(FCBNVal)); }
    void reset()
    {
        vecMuBeta.reset();
        vecSigmaSqr.reset();
        setBarX.reset();
        setY.reset();
    }
    ~FCBN() { reset(); }
};

FCBN BNTrain(set<vect> &setInput, double dBeta = 0, double dGamma = 1, double dEpsilon = 1e-8)
{
    FCBN BNOutput;
    // Average, miu
    BNOutput.vecMuBeta = (1.0 / setInput.size()) * setInput.sum();
    // Variance, sigma square
    for(auto i=0; i<setInput.size(); ++i)
        if(BNOutput.vecSigmaSqr.is_matrix()) BNOutput.vecSigmaSqr += (setInput[i] - BNOutput.vecMuBeta).elem_cal_opt(2, MATRIX_ELEM_POW);
        else BNOutput.vecSigmaSqr = (setInput[i] - BNOutput.vecMuBeta).elem_cal_opt(2, MATRIX_ELEM_POW);
    BNOutput.vecSigmaSqr *= (1.0 / setInput.size());
    BNOutput.setBarX.init(setInput.size());
    BNOutput.setY.init(setInput.size());
    // Normalize, bar x
    for(auto i=0; i<setInput.size(); ++i) BNOutput.setBarX[i] = (setInput[i] - BNOutput.vecMuBeta).elem_cal_opt(DIV_DOM(BNOutput.vecSigmaSqr, dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
    // Scale shift, y
    for(auto i=0; i<setInput.size(); ++i) BNOutput.setY[i] = (dGamma * BNOutput.setBarX[i]).broadcast_add(dBeta);
    return BNOutput;
}

set<vect> BNGradLossToInput(FCBN &FCBNOutput, set<vect> &setInput, set<vect> &setGradLossToOutput, double dGamma, double dEpsilon = 1e-8)
{
    // Operation value
    auto vecDmtSigmaSqr = DIV_DOM(FCBNOutput.vecSigmaSqr, dEpsilon);
    auto vecDmtSigma = vecDmtSigmaSqr.elem_cal_opt(0.5, MATRIX_ELEM_POW);
    // Gradient loss to normalized output, bar x
    set<vect> setGradLossToBarX(setInput.size());
    for(auto i=0; i<setInput.size(); ++i) setGradLossToBarX[i] = setGradLossToOutput[i] * dGamma;
    // Gradient loss to variance, square-powered sigma
    vect vecGradLossToSigmaSqr(vecDmtSigma.LN_CNT, vecDmtSigma.COL_CNT);
    for(auto i=0; i<setInput.size(); ++i) vecGradLossToSigmaSqr += setGradLossToBarX[i].elem_cal_opt((setInput[i]-FCBNOutput.vecMuBeta),  MATRIX_ELEM_MULT).elem_cal_opt(vecDmtSigmaSqr.elem_cal_opt(1.5,  MATRIX_ELEM_POW),  MATRIX_ELEM_DIV);
    vecGradLossToSigmaSqr *= (-0.5);
    // Gradient loss to average, mubeta
    vect vecDistanceSum = vect(vecDmtSigma.LN_CNT, vecDmtSigma.COL_CNT);
    for(auto i=0; i<setInput.size(); ++i) vecDistanceSum += (setInput[i] - FCBNOutput.vecMuBeta);
    vect vecGradMuBeta = (-1) * setGradLossToBarX.sum().elem_cal_opt(vecDmtSigma, MATRIX_ELEM_DIV) + ((-2.0) / setInput.size()) * vecGradLossToSigmaSqr.elem_cal_opt(vecDistanceSum, MATRIX_ELEM_MULT);
    // Gradient loss to input, x
    set<vect> setGradLossToInput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
        setGradLossToInput[i] = setGradLossToBarX[i].elem_cal_opt(vecDmtSigma, MATRIX_ELEM_DIV) + ((2.0) / setInput.size()) * vecGradLossToSigmaSqr.elem_cal_opt((setInput[i]-FCBNOutput.vecMuBeta), MATRIX_ELEM_MULT) + ((1.0) / setInput.size()) * vecGradMuBeta;
    return setGradLossToInput;
}

double BNGradLossToScale(set<vect> &setGradLossToOutput, FCBN &FCBNOutput)
{
    double dGrad = 0;
    for(auto i=0; i<setGradLossToOutput.size(); ++i) for(auto j=0; j<setGradLossToOutput[i].ELEM_CNT; ++j) dGrad += setGradLossToOutput[i].pos_idx(j) * FCBNOutput.setBarX[i].pos_idx(j);
    return dGrad;
}

double BNGradLossToShift(set<vect> &setGradLossToOutput)
{
    double dGrad = 0;
    for(auto i=0; i<setGradLossToOutput.size(); ++i) for(auto j=0; j<setGradLossToOutput[i].ELEM_CNT; ++j) dGrad += setGradLossToOutput[i].pos_idx(j);
    return dGrad;
}

void BNDeduceInit(BN_EXP_VAR &BNData, uint64_t iBatchCnt, uint64_t iBatchSize)
{
    /**
     * Expectation Average, Expectation MiuBeta
     * Variance mini-batch variance, Variance SigmaSqr
     */
    if(iBatchCnt)
    {
        BNData.vecExp = BNData.vecExp.elem_cal_opt(iBatchCnt, MATRIX_ELEM_DIV);
        BNData.vecVar = BNData.vecVar.elem_cal_opt(iBatchCnt, MATRIX_ELEM_DIV);
        if(iBatchSize > 1) BNData.vecVar *= (iBatchSize / (iBatchSize - 1.0));
    }
}

vect BNDeduce(vect &vecInput, double dBeta, double dGamma, BN_EXP_VAR &BNData, double dEpsilon = 1e-8)
{
    // Normalize
    auto vecNorm = vecInput - BNData.vecExp;
    auto vecVar = DIV_DOM(BNData.vecVar, dEpsilon);
    auto vecBarX = vecNorm.elem_cal_opt(vecVar.elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
    return (dGamma * vecBarX).broadcast_add(dBeta);
}

FC_END
ADA_BEGIN

/* @brief Learn rate adeaptive decay
 * @param   dLearnRate  [Input] Learn rate
 * @param   iGlobalStep [Input] Global step, batch accumulation, refresh for a new epoch
 * @param   dDecay      [Input] Decay step
 * @return  New learn rate for next round
 */
double AdaExpDecayLearnRate(double dLearnRate, uint64_t iGlobalStep, double dDecay = 0.9) { return dLearnRate * std::exp((-1) * dDecay / iGlobalStep); }

struct AdaDeltaVect
{
private:
    vect vecExpGrad;
    vect vecExpDelta;
public:
    double dRho = 0.95;
    double dEpsilon = 1e-8;
    AdaDeltaVect(){}
    AdaDeltaVect(AdaDeltaVect &advSrc) {*this = advSrc;}
    AdaDeltaVect(AdaDeltaVect &&advSrc) {*this = std::move(advSrc);}
    AdaDeltaVect(uint64_t nSizeLnCnt, uint64_t nSizeColCnt, double dRho = 0.95, double dEpsilon = 1e-8)
    {
        vecExpGrad = vect(nSizeLnCnt, nSizeColCnt);
        vecExpDelta = vect(nSizeLnCnt, nSizeColCnt);
        this->dRho = dRho;
        this->dEpsilon = dEpsilon;
    }
    void operator=(AdaDeltaVect &advSrc)
    {
        dRho = advSrc.dRho;
        dEpsilon = advSrc.dEpsilon;
        vecExpDelta = advSrc.vecExpDelta;
        vecExpGrad = advSrc.vecExpGrad;
    }
    void operator=(AdaDeltaVect &&advSrc)
    {
        dRho = advSrc.dRho;
        dEpsilon = advSrc.dEpsilon;
        vecExpDelta = std::move(advSrc.vecExpDelta);
        vecExpGrad = std::move(advSrc.vecExpGrad);
    }
    vect Delta(vect &vecCurrGrad)
    {
        if(!vecExpGrad.is_matrix()) vecExpGrad = vect(vecCurrGrad.LN_CNT, vecCurrGrad.COL_CNT);
        if(!vecExpDelta.is_matrix()) vecExpDelta = vecExpGrad;
        vecExpGrad = dRho * vecExpGrad + (1 - dRho) * vecCurrGrad.elem_cal_opt(2, MATRIX_ELEM_POW);
        auto vecRMSPreDelta = DIV_DOM(vecExpDelta, dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW),
            vecRMSGrad = DIV_DOM(vecExpGrad, dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW);
        auto vecCurrDelta = vecRMSPreDelta.elem_cal_opt(vecRMSGrad, MATRIX_ELEM_DIV).elem_cal_opt(vecCurrGrad, MATRIX_ELEM_MULT);
        vecExpDelta = dRho * vecExpDelta + (1 - dRho) * vecCurrDelta.elem_cal_opt(2, MATRIX_ELEM_POW);
        return vecCurrDelta;
    }
    void Reset() { vecExpDelta.reset(); vecExpGrad.reset(); }
    ~AdaDeltaVect() { Reset(); }
};

struct AdaDeltaVal
{
private:
    double dExpGrad = 0;
    double dExpDelta = 0;
public:
    double dRho = 0.95;
    double dEpsilon = 1e-8;
    AdaDeltaVal(double dRhoVal = 0.95, double dEpsilonVal = 1e-8) : dRho(dRhoVal), dEpsilon(dEpsilonVal) {}
    AdaDeltaVal(AdaDeltaVal &advSrc) { *this = advSrc; }
    void operator=(AdaDeltaVal &advSrc)
    {
        dExpGrad = advSrc.dExpGrad;
        dExpDelta = advSrc.dExpDelta;
        dRho = advSrc.dRho;
        dEpsilon = advSrc.dEpsilon;
    }
    double Delta(double dCurrGrad)
    {
        dExpGrad = dRho * dExpGrad + (1 - dRho) * dCurrGrad * dCurrGrad;
        if(!dExpDelta) dExpDelta += dEpsilon;
        if(!dExpGrad) dExpGrad += dEpsilon;
        auto dRMSPreDelta = std::pow(dExpDelta, 0.5), dRMSCurrGrad = std::pow(dExpGrad, 0.5);
        auto vecCurrDelta = (dRMSPreDelta / dRMSCurrGrad) * dCurrGrad;
        dExpDelta = dRho * dExpDelta + (1 - dRho) * vecCurrDelta * vecCurrDelta;
        return vecCurrDelta;
    }
    void Reset() { dExpGrad = 0; dExpDelta = 0; }
};

struct AdaNesterovVect
{
private:
    vect vecVelocity;
    double dRho = 0.9;
public:
    AdaNesterovVect() {}
    AdaNesterovVect(AdaNesterovVect &anvSrc) { *this = anvSrc; }
    AdaNesterovVect(AdaNesterovVect &&anvSrc) { *this = std::move(anvSrc); }
    void operator=(AdaNesterovVect &anvSrc) { vecVelocity = anvSrc.vecVelocity; dRho = anvSrc.dRho; }
    void operator=(AdaNesterovVect &&anvSrc) { vecVelocity = std::move(anvSrc.vecVelocity); dRho = anvSrc.dRho; }

    vect NesterovWeight(vect &vecWeight)
    {
        if(vecVelocity.is_matrix()) return vecWeight + dRho * vecVelocity;
        else return vecWeight;
    }
    vect NesterovMomentum(vect &vecGrad, double dLearnRate)
    {
        if(vecVelocity.is_matrix()) vecVelocity = dRho * vecVelocity - dLearnRate * vecGrad;
        else vecVelocity = (-1) * dLearnRate * vecGrad;
        return (-1) * vecVelocity;
    }

    void Reset() { vecVelocity.reset(); dRho = 0.9; }
    ~AdaNesterovVect() { Reset(); }
};

struct AdaNesterovVal
{
private:
    double dVelocity = 0;
    double dRho = 0.9;
public:
    AdaNesterovVal() {}
    AdaNesterovVal(AdaNesterovVal &anvSrc) { *this = anvSrc; }
    void operator=(AdaNesterovVal &anvSrc) { dRho = anvSrc.dRho; dVelocity = anvSrc.dVelocity; }

    double NesterovWeight(double dWeight) { return dWeight +  dRho * dVelocity; }

    double NesterovMomentum(double dGrad, double dLearnRate)
    {
        dVelocity = dRho * dVelocity - dLearnRate * dGrad;
        return (-1) * dVelocity;
    }

    void Reset() { dVelocity = 0; dRho = 0.9; }
};

ADA_END

FC_BEGIN

vect AdaDeltaUpdateWeight(vect &vecWeight, vect &vecGradLossToWeight, ada::AdaDeltaVect &advCurrLayerDelta)
{
    auto vecCurrDelta = advCurrLayerDelta.Delta(vecGradLossToWeight);
    if(vecCurrDelta.is_matrix()) return vecWeight - vecCurrDelta;
    else return blank_vect;
}

double BNAdaDeltaUpdateScaleShift(double dGammaBeta, double dGradLossToScaleShift, ada::AdaDeltaVal &advCurrDelta) { return dGammaBeta - advCurrDelta.Delta(dGradLossToScaleShift); }


vect AdaNesterovUpdateWeight(vect &vecWeight, vect &vecGradLossToWeight, double dLearnRate, ada::AdaNesterovVect &anvCurrLayerMomt)
{
    auto vecCurrMomt = anvCurrLayerMomt.NesterovMomentum(vecGradLossToWeight, dLearnRate);
    if(vecCurrMomt.is_matrix()) return vecWeight - vecCurrMomt;
    else return blank_vect;
}

double BNAdaNesterovUpdateScaleShift(double dGammaBeta, double dGradLossToScaleShift, double dLearnRate, ada::AdaNesterovVal &anvCurrMomt) { return dGammaBeta - anvCurrMomt.NesterovMomentum(dGradLossToScaleShift, dLearnRate); }

FC_END

CONV_BEGIN

tensor AdaDeltaUpdateKernel(tensor &tenKernel, tensor &tenGradLossToKernel, ada::ada_tensor<ada::AdaDeltaVect> &advCurrDelta)
{
    if(tenKernel.size() == tenGradLossToKernel.size())
    {
        tensor tenUpdatedKernel(tenKernel.size());
        for(auto i=0; i<tenKernel.size(); ++i)
            if(tenKernel[i].size() == tenGradLossToKernel[i].size())
            {
                tenUpdatedKernel[i].init(tenKernel[i].size());
                for(auto j=0; j<tenKernel[i].size(); ++j)
                {
                    tenUpdatedKernel[i][j] = tenKernel[i][j] - advCurrDelta[i][j].Delta(tenGradLossToKernel[i][j]);
                    if(!tenUpdatedKernel[i][j].is_matrix()) return blank_tensor;
                }
            }
            else return blank_tensor;
        return tenUpdatedKernel;
    }
    else return blank_tensor;
}

vect BNAdaDeltaUpdateScaleShift(vect &vecGammaBeta, vect &vecGradLossToScaleShift, ada::AdaDeltaVect &advCurrDelta)
{
    if(vecGammaBeta.shape_valid(vecGradLossToScaleShift)) return vecGammaBeta - advCurrDelta.Delta(vecGradLossToScaleShift);
    else return blank_vect;
}

tensor AdaNesterovUpdateKernel(tensor &tenKernel, tensor &tenGradLossToKernel, double dLearnRate, ada::ada_tensor<ada::AdaNesterovVect> &anvCurrMomt)
{
    if(tenKernel.size() == tenGradLossToKernel.size())
    {
        tensor tenUpdatedKernel(tenKernel.size());
        for(auto i=0; i<tenKernel.size(); ++i)
            if(tenKernel[i].size() == tenGradLossToKernel[i].size())
            {
                tenUpdatedKernel[i].init(tenKernel[i].size());
                for(auto j=0; j<tenKernel[i].size(); ++j)
                {
                    tenUpdatedKernel[i][j] = tenKernel[i][j] - anvCurrMomt[i][j].NesterovMomentum(tenGradLossToKernel[i][j], dLearnRate);
                    if(!tenUpdatedKernel[i][j].is_matrix()) return blank_tensor;
                }
            }
            else return blank_tensor;
        return tenUpdatedKernel;
    }
    else return blank_tensor;
}

vect BNAdaNesterovUpdateScaleShift(vect &vecGammaBeta, vect &vecGradLossToScaleShift, double dLearnRate, ada::AdaNesterovVect &anvCurrMomt)
{
    if(vecGammaBeta.shape_valid(vecGradLossToScaleShift)) return vecGammaBeta - anvCurrMomt.NesterovMomentum(vecGradLossToScaleShift, dLearnRate);
    else return blank_vect;
}

CONV_END
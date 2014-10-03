//
// MATLAB Compiler: 5.0 (R2013b)
// Date: Tue Sep 30 14:38:42 2014
// Arguments: "-B" "macro_default" "-v" "-W" "cpplib:libmodelBasedSmoothing2"
// "-T" "link:lib" "modelBasedSmoothing_wARGS" 
//

#include <stdio.h>
#define EXPORTING_libmodelBasedSmoothing2 1
#include "libmodelBasedSmoothing2.h"

static HMCRINSTANCE _mcr_inst = NULL;


#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultPrintHandler(const char *s)
{
  return mclWrite(1 /* stdout */, s, sizeof(char)*strlen(s));
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultErrorHandler(const char *s)
{
  int written = 0;
  size_t len = 0;
  len = strlen(s);
  written = mclWrite(2 /* stderr */, s, sizeof(char)*len);
  if (len > 0 && s[ len-1 ] != '\n')
    written += mclWrite(2 /* stderr */, "\n", sizeof(char));
  return written;
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_libmodelBasedSmoothing2_C_API
#define LIB_libmodelBasedSmoothing2_C_API /* No special import/export declaration */
#endif

LIB_libmodelBasedSmoothing2_C_API 
bool MW_CALL_CONV libmodelBasedSmoothing2InitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
    int bResult = 0;
  if (_mcr_inst != NULL)
    return true;
  if (!mclmcrInitialize())
    return false;
    {
        mclCtfStream ctfStream = 
            mclGetEmbeddedCtfStream((void 
                                                     *)(libmodelBasedSmoothing2InitializeWithHandlers));
        if (ctfStream) {
            bResult = mclInitializeComponentInstanceEmbedded(   &_mcr_inst,
                                                                error_handler, 
                                                                print_handler,
                                                                ctfStream);
            mclDestroyStream(ctfStream);
        } else {
            bResult = 0;
        }
    }  
    if (!bResult)
    return false;
  return true;
}

LIB_libmodelBasedSmoothing2_C_API 
bool MW_CALL_CONV libmodelBasedSmoothing2Initialize(void)
{
  return libmodelBasedSmoothing2InitializeWithHandlers(mclDefaultErrorHandler, 
                                                       mclDefaultPrintHandler);
}

LIB_libmodelBasedSmoothing2_C_API 
void MW_CALL_CONV libmodelBasedSmoothing2Terminate(void)
{
  if (_mcr_inst != NULL)
    mclTerminateInstance(&_mcr_inst);
}

LIB_libmodelBasedSmoothing2_C_API 
void MW_CALL_CONV libmodelBasedSmoothing2PrintStackTrace(void) 
{
  char** stackTrace;
  int stackDepth = mclGetStackTrace(&stackTrace);
  int i;
  for(i=0; i<stackDepth; i++)
  {
    mclWrite(2 /* stderr */, stackTrace[i], sizeof(char)*strlen(stackTrace[i]));
    mclWrite(2 /* stderr */, "\n", sizeof(char)*strlen("\n"));
  }
  mclFreeStackTrace(&stackTrace, stackDepth);
}


LIB_libmodelBasedSmoothing2_C_API 
bool MW_CALL_CONV mlxModelBasedSmoothing_wARGS(int nlhs, mxArray *plhs[], int nrhs, 
                                               mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "modelBasedSmoothing_wARGS", nlhs, plhs, nrhs, prhs);
}

LIB_libmodelBasedSmoothing2_CPP_API 
void MW_CALL_CONV modelBasedSmoothing_wARGS(int nargout, mwArray& smoothProbMap, const 
                                            mwArray& probMap, const mwArray& 
                                            probMapShape, const mwArray& 
                                            numCentroidsUsed, const mwArray& numFits, 
                                            const mwArray& sampling)
{
  mclcppMlfFeval(_mcr_inst, "modelBasedSmoothing_wARGS", nargout, 1, 5, &smoothProbMap, &probMap, &probMapShape, &numCentroidsUsed, &numFits, &sampling);
}


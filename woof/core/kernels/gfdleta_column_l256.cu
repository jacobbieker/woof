GD_DEVICE void gd_column_one(GDTable *table,float *f,int *ints,GDScratch *work,int longwave=1,int shortwave=1) {
 GDClock clock={0.0f,0.0f,0.0f,0.0f};
 ints[5]=longwave;ints[6]=longwave?0:shortwave;
 etara_(f+0,f+1,f+258,f+515,f+772,f+1029,f+1286,f+1287,f+1544,f+1801,f+2058,f+2315,f+2572,f+2829,f+3086,f+3343,f+3600,f+3601,f+3602,f+3603,f+3604,f+3605,f+3606,f+3607,f+3608,f+3609,f+3610,f+3611,f+3612,f+3613,f+3614,f+3615,f+3616,f+3617,f+3618,f+3619,ints+0,ints+1,ints+2,f+3620,f+3621,ints+3,ints+4,ints+5,ints+6,f+3622,f+3623,f+3624,f+3625,ints+7,f+3626,ints+8,ints+9,ints+10,ints+11,ints+12,ints+13,ints+14,ints+15,ints+16,ints+17,ints+18,ints+19,ints+20,ints+21,ints+22,ints+23,ints+24,ints+25,ints+26,table,&clock,work);
 if(!(longwave&&shortwave))return;
 ints[6]=1;
 etara_(f+0,f+1,f+258,f+515,f+772,f+1029,f+1286,f+1287,f+1544,f+1801,f+2058,f+2315,f+2572,f+2829,f+3086,f+3343,f+3600,f+3601,f+3602,f+3603,f+3604,f+3605,f+3606,f+3607,f+3608,f+3609,f+3610,f+3611,f+3612,f+3613,f+3614,f+3615,f+3616,f+3617,f+3618,f+3619,ints+0,ints+1,ints+2,f+3620,f+3621,ints+3,ints+4,ints+5,ints+6,f+3622,f+3623,f+3624,f+3625,ints+7,f+3626,ints+8,ints+9,ints+10,ints+11,ints+12,ints+13,ints+14,ints+15,ints+16,ints+17,ints+18,ints+19,ints+20,ints+21,ints+22,ints+23,ints+24,ints+25,ints+26,table,&clock,work);
}
#ifndef GD_HOST_DIAGNOSTIC
extern "C" __global__ void gd_radiation_columns(GDTable *table,float *floats,int *integers,GDScratch *work,int ncol) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c<ncol)gd_column_one(table,floats+c*3627,integers+c*27,work+c);
}

extern "C" __global__ void gd_radiation_columns_pooled(GDTable *table,float *floats,int *integers,GDScratch *work,int ncol,int slots,int longwave,int shortwave) {
 int slot=blockIdx.x*blockDim.x+threadIdx.x;
 if(slot>=slots)return;
 for(int c=slot;c<ncol;c+=slots)
  gd_column_one(table,floats+c*3627,integers+c*27,work+slot,longwave,shortwave);
}
extern "C" __global__ void gd_radiation_call(GDTable *table,float *floats,int *integers,GDScratch *scratch,GDClock *clocks,int ncol) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c>=ncol)return;
 float *f=floats+c*3627;int *ints=integers+c*27;
 GDScratch *work=scratch+c;GDClock &clock=clocks[c];
 etara_(f+0,f+1,f+258,f+515,f+772,f+1029,f+1286,f+1287,f+1544,f+1801,f+2058,f+2315,f+2572,f+2829,f+3086,f+3343,f+3600,f+3601,f+3602,f+3603,f+3604,f+3605,f+3606,f+3607,f+3608,f+3609,f+3610,f+3611,f+3612,f+3613,f+3614,f+3615,f+3616,f+3617,f+3618,f+3619,ints+0,ints+1,ints+2,f+3620,f+3621,ints+3,ints+4,ints+5,ints+6,f+3622,f+3623,f+3624,f+3625,ints+7,f+3626,ints+8,ints+9,ints+10,ints+11,ints+12,ints+13,ints+14,ints+15,ints+16,ints+17,ints+18,ints+19,ints+20,ints+21,ints+22,ints+23,ints+24,ints+25,ints+26,table,&clock,work);
}
#endif

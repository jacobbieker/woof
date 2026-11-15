/* WRF observation nudging (obs_nudge_opt = 1) on hex cell columns.
 *
 * Compiled after the shared pieces of woof/fddaobs/kernels.cu and forecast.cu
 * (Frame, Grid, bound, nintf, timewt, Controls, fddaobs_vif, fddaobs_publish,
 * fddaobs_clock), which woof.hex.obs_nudging cuts from those files unchanged.
 * The two kernels below are fddaobs_errob and the cell schedule of
 * fddaobs_nudob_rows (WRF v4.7.1 module_fddaobs_rtfdda.F ERROB and NUDOB),
 * expression for expression, with the mesh in place of the ARW grid:
 *
 *  - Hex arrays are [level][cell], the WRF (i,k,j) layout of a one-row grid
 *    (i = cell + 1, j = 1), so Grid::at and the vertical loops are WRF's.
 *  - ERROB's bilinear interpolation in (i, j) is the barycentric
 *    interpolation on the Delaunay triangle (three cell centres) holding the
 *    report; u, v and mass share the cell centre, so the three staggered
 *    passes are one.  The surface pressure of the wind innovations (WRF's
 *    u/v-point base pressure) is the base pressure at the report.
 *  - NUDOB's planar grid distance rsq = (j-rb)^2 + (i-ra)^2, in grid lengths,
 *    is the great-circle distance squared, in units of `dx` metres, from a
 *    per-cell list of the reports within reach (a cell KD-tree on the host).
 *    WRF's i/j bounding boxes and its out-of-domain screen select a superset
 *    of the cells with a nonzero weight; the list holds exactly those.
 *  - Coupled tendencies (mu/msf, c1h mu + c2h) become the uncoupled rates
 *    coefficient * weight * tfaci: the sphere has no map factor, and
 *    woof.hex.nudging couples them at the IAU seat.
 */
extern "C" __global__ void hex_fddaobs_errob(
 const float*ub,const float*vb,const float*tb,const float*qv,const float*pb,const float*pp,
 const float*z,const float*ur,const float*vr,const float*tr,const int*kpbl,
 const int*corner,const float*bary,const int*near,float*rko,float*obs,float*err,
 int count,int nvar,int nerr,int wind,int temp,int mois,float t0,float rcp,Grid g,const Frame*frame){
 int n=int(blockIdx.x*blockDim.x+threadIdx.x);if(n>=count)return;
 if(frame&&(!frame->enabled||!frame->update||n<frame->first||n>=frame->last))return;
 float*e=err+n*nerr;float*o=obs+n*nvar;for(int v=0;v<nerr;v++)e[v]=0.f;
 int kend=min(g.kte,g.kde-1);
 int c0=corner[3*n],c1=corner[3*n+1],c2=corner[3*n+2];
 float w0=bary[3*n],w1=bary[3*n+1],w2=bary[3*n+2];
 #define TRI(a,k) (w0*(a)[g.at(c0,k,1)]+w1*(a)[g.at(c1,k,1)]+w2*(a)[g.at(c2,k,1)])
 #define TRI2(a) (w0*(a)[g.xy(c0,1)]+w1*(a)[g.xy(c1,1)]+w2*(a)[g.xy(c2,1)])
 if(fabsf(rko[n]+99.f)<1.f){
  float pob=o[4];
  if(pob==-888888.f&&o[5]>-800000.f){
   int klo=g.kts;
   for(int k=g.kts+1;k<=kend;k++){klo=k-1;if(o[5]<=TRI(z,k))break;}
   float zlo=TRI(z,klo),zhi=TRI(z,klo+1);
   float lp0=logf(.001f*TRI(pb,klo+1)+.001f*TRI(pp,klo+1));
   float lp1=logf(.001f*TRI(pb,klo)+.001f*TRI(pp,klo));
   float lp=o[5]<=zlo?lp1:o[5]>=zhi?lp0:lp0+(lp1-lp0)*((zhi-o[5])/(zhi-zlo));pob=expf(lp);
  }
  if(pob>-800000.f){
   int kb=1;for(int k=kend-1;k>=1;k--){kb=k;if(pob<=.001f*TRI(pb,k)+.001f*TRI(pp,k))break;}
   float hi=.001f*TRI(pb,kb+1),hp=.001f*TRI(pp,kb+1);
   rko[n]=max(float(kb+1)-(pob-hi-hp)/(.001f*TRI(pb,kb)+.001f*TRI(pp,kb)-hi-hp),1.f);
  }
  if(o[4]==-888888.f&&o[5]>-800000.f)o[4]=pob;
 }
 int ic=int(rko[n]);float dz=rko[n]-float(ic);
 float dummy=0.f;
 if(rko[n]<1.f)dummy+=99999.f;
 for(int v=0;v<=4;v++)e[v]+=dummy;
 e[8]=rko[n];
 int k=min(kend,ic),kp=min(k+1,kend);
 if(wind==1){
  float ratio=1.f;if(rko[n]==1.f){ratio=TRI2(ur);}
  if(fabsf(ratio)<1.e-3f)ratio=1.f;
  e[0]=e[0]+ratio*o[0]-((1.f-dz)*TRI(ub,k)+dz*TRI(ub,kp));
  ratio=1.f;if(rko[n]==1.f){ratio=TRI2(vr);}
  if(fabsf(ratio)<1.e-3f)ratio=1.f;
  e[1]=e[1]+ratio*o[1]-((1.f-dz)*TRI(vb,k)+dz*TRI(vb,kp));
  e[6]=.001f*TRI(pb,1);e[7]=e[6];
 }
 if(temp==1||mois==1){
  float ratio=rko[n]==1.f?TRI2(tr):1.f;if(fabsf(ratio)<1.e-3f)ratio=1.f;
  float theta=-888888.f;if(o[2]>-800000.f&&o[4]>-800000.f)theta=o[2]*powf(100.f/o[4],rcp)-t0;
  e[2]=e[2]+ratio*theta-((1.f-dz)*TRI(tb,k)+dz*TRI(tb,kp));
  float modelq=(1.f-dz)*TRI(qv,k)+dz*TRI(qv,kp);
  e[3]=e[3]+o[3]-modelq;e[9]=e[9]+modelq;e[5]=.001f*TRI(pb,1);
 }
 e[4]=float(kpbl[g.xy(near[n],1)]);
 if(fabsf(e[2])>20.f&&e[2]>-800000.f){for(int v=0;v<4;v++){e[v]=-888888.f;o[v]=-888888.f;}}
 #undef TRI
 #undef TRI2
}
/* One thread per (cell, level): fddaobs_nudob_rows' cell schedule.  The
 * reports a cell can feel are reach[start[i-1] .. start[i]) (report start
 * rows, ascending: WRF's observation order) with their squared distances. */
extern "C" __global__ void hex_fddaobs_nudob(
 float*aten,float*savwt,const float*mask,
 const int*start,const int*reach,const float*reach_rsq,
 const float*rko,const float*timeob,const float*obs,const float*err,
 const float*nlevs,const float*lev,const float*pb,const float*pp,const int*kpbl,
 const float*vih1,const float*vih2,const float*terrain,const float*z,
 int nvar,int nerr,int ivar,Controls c,Grid g,const Frame*frame,const int*surface_top){
 int i=int(blockIdx.x*blockDim.x+threadIdx.x)+g.its;
 int kcell=int(blockIdx.y*blockDim.y+threadIdx.y)+g.kts;
 if(i>g.ite||kcell>g.kte)return;
 int j=1,kbeg=kcell,kend=kcell;
 float local_wt=0.f,local_w2=0.f;
 // WRF adds nothing outside the window; this tendency slot is written every step.
 if(frame){if(!frame->enabled||frame->first==frame->last){aten[g.at(i,kcell,j)]=0.f;return;} c.ktau=frame->ktau;c.xtime=frame->xtime;c.tfaci=frame->tfaci;c.dtmin=frame->dtmin;}
 bool update=frame?frame->update:(c.ktau%c.interval==0||(c.restart&&c.ktau==c.ktaur));
 #define A(i,k) g.at(i,k,j)
 #define XY(i) g.xy(i,j)
 #define E(v,n) err[(n)*nerr+(v)-1]
 #define O(v,n) obs[(n)*nvar+(v)-1]
 if(update){
  float rindx=c.rinxy*1000.f/c.dx;
  for(int r=start[i-g.its];r<start[i-g.its+1];r++){
   int n=reach[r];
   if(frame&&(n<frame->first||n>=frame->last))continue;
   int ns=int(nlevs[n]-lev[n])+1;
   float pob=O(5,n);
   if(fabsf(E(ivar,n))>9.e4f&&ns==1)continue;
   int kob=bound(nintf(rko[n]+.45f),1,g.kte);
   float erf=E(ivar,n),psurf=E(ivar>=3?6:ivar==1?7:8,n);
   bool surface=kob==1&&nlevs[n]<1.5f;
   float tw=timewt(c.xtime,c.dtmin,c.window,surface?c.sfcfact:1.f,timeob[n]);
   float radius=c.sfcfacr;
   if(!surface){float slope=(c.rinfmn-c.rinfmx)/(psurf-c.pfree);radius=min(max(slope*pob+c.rinfmx-slope*c.pfree,c.rinfmn),c.rinfmx);if(ns>1)radius=c.rinfmx;}
   if(surface&&!c.nudge_pbl)continue;
   float rsq=reach_rsq[r],ris;
   if(surface)ris=rindx*rindx*c.sfcfacr*c.sfcfacr;else ris=rindx*radius*rindx*radius;
   float wh;
   if(surface&&c.horiz==1){float dprim=sqrtf(rsq);float d=dprim+rindx*c.dcon*fabsf(psurf-.001f*pb[A(i,1)]);float dsq=d*d;wh=max(0.f,(ris-dsq)/(ris+dsq));}
   else{wh=(ris-rsq)/(ris+rsq);if(surface){float scratch=fabsf(psurf-.001f*pb[A(i,1)])*c.dcon;float pdfac=1.f-min(1.f,scratch);wh=wh*pdfac;}wh=max(0.f,wh);}
   if(surface){
    int komax=g.kte;
    if(c.vert==1){komax=max(3,kpbl[XY(i)]);if(kpbl[XY(i)]>25)komax=3;}
    float dzr=1.f/max(1.f,vih2[XY(i)]-vih1[XY(i)]);
    if(c.vert!=1)komax=min(komax,surface_top[XY(i)]);
    for(int k=max(c.vert==1?1:g.kts,kbeg);k<=min(komax,kend);k++){
     float wv=c.vert==1?float(komax-k+1)/float(komax):max(0.f,min(1.f,1.f+(vih1[XY(i)]-z[A(i,k)]+terrain[XY(i)])*dzr));
     if(c.vert!=1&&wv<=0.f)break;
     if(k!=kcell)continue;
     local_wt=local_wt+tw*wv*wh;
     local_w2=local_w2+tw*tw*wh*wh*wv*wv*erf;
    }
   }else if(ns==1){
    int komin=1,komax=g.kte;
    for(int k=g.kte;k>=1;k--)if(.001f*(pb[A(i,k)]+pp[A(i,k)])>=pob+7.5f){komin=k;break;}
    for(int k=3;k<=g.kte;k++)if(.001f*(pb[A(i,k)]+pp[A(i,k)])<=pob-7.5f){komax=k;break;}
    if(kpbl[XY(i)]<=komax&&kpbl[XY(i)]>=komin){
     int kobs=1;for(int k=komin;k<=komax;k++)if(pob>.001f*(pb[A(i,k)]+pp[A(i,k)])){kobs=k;break;}
     if(kobs>int(E(5,n)))komin=max(kobs,komin);else komax=min(kpbl[XY(i)],komax);
    }
    if(c.nudge_pbl||komin>=kpbl[XY(i)])for(int k=max(komin,kbeg);k<=min(komax,kend);k++){
     float pijk=.001f*(pb[A(i,k)]+pp[A(i,k)]),res=erf,wv=max(1.f-fabsf(pijk-pob)/7.5f,0.f);
     local_wt=local_wt+tw*wh*wv;
     local_w2=local_w2+tw*tw*wh*wh*res*wv*wv;
    }
   }else{
    for(int k=min(g.kte-2,kend);k>=kbeg;k--){
     float pijk=.001f*(pb[A(i,k)]+pp[A(i,k)]);if(pijk>O(5,n)||pijk<=O(5,n+ns-1))continue;
     float slope=(c.rinfmn-c.rinfmx)/(.001f*pb[A(i,1)]-c.pfree);
     float rf=min(max(slope*pijk+c.rinfmx-slope*c.pfree,c.rinfmn),c.rinfmx);
     ris=rindx*rf*rindx*rf;wh=max(0.f,(ris-rsq)/(ris+rsq));
     int nhi=-1,nlo=-1;float phi=0.f,plo=0.f,above=c.maxsnd_gap,below=c.maxsnd_gap;
     for(int nn=1;nn<ns;nn++)if(O(ivar,n+nn)>-800000.f&&O(5,n+nn)>-800000.f){phi=O(5,n+nn);above=fabsf(phi-pijk);if(phi<=pijk&&above<=c.maxsnd_gap){nhi=n+nn;break;}}
     if(nhi<0)continue;
     for(int nn=nhi-1;nn>=n;nn--)if(O(ivar,nn)>-800000.f&&O(5,nn)>-800000.f){plo=O(5,nn);below=fabsf(plo-pijk);if(plo>=pijk&&below<=c.maxsnd_gap){nlo=nn;break;}}
     if(nlo<0||above+below>c.maxsnd_gap)continue;
     float pdiff=fabsf(phi-plo)<.000001f?0.f:logf(pijk/plo)/logf(phi/plo);
     float res=E(ivar,nlo)+(E(ivar,nhi)-E(ivar,nlo))*pdiff;
     if(c.nudge_pbl||k>kpbl[XY(i)]){float wv=1.f;local_wt=local_wt+tw*wh*wv;local_w2=local_w2+tw*tw*wh*wh*res*wv*wv;}
    }
   }
  }
  if(local_wt==0.f){local_w2=0.f;local_wt=1.f;}
  savwt[(A(i,kcell))*nvar+ivar-1]=local_w2/local_wt;
 }
 float w=savwt[A(i,kcell)*nvar+ivar-1];
 aten[A(i,kcell)]=mask[XY(i)]!=0.f?c.coefficient*w*c.tfaci*1.f*1.f:0.f;
 #undef A
 #undef XY
 #undef E
 #undef O
}
/* The surface ratios of fddaobs_pack (lowest model level over the 10 m wind
 * and the 2 m potential temperature) on the cell columns, refreshed on
 * WRF's PBL clock (ktau == 0 or (ktau+1) % stepbl == 0). */
extern "C" __global__ void hex_fddaobs_ratios(
 const float*u,const float*v,const float*theta,const float*p,
 const float*u10,const float*v10,const float*th2,float*ur,float*vr,float*tr,
 int nc,int sfclay,int stepbl,const Frame*frame){
 int h=int(blockIdx.x*blockDim.x+threadIdx.x);if(h>=nc)return;
 if(!(frame->ktau==0||(frame->ktau+1)%stepbl==0))return;
 float ua=u[h],va=v[h];
 ur[h]=!sfclay?0.f:fabsf(u10[h])>1.e-10f?ua/u10[h]:1.2f;
 vr[h]=!sfclay?0.f:fabsf(v10[h])>1.e-10f?va/v10[h]:1.2f;
 float temp=theta[h]*powf(p[h]/100000.f,287.f/1004.5f);
 tr[h]=!sfclay?0.f:temp*powf(100000.f*.001f/(p[h]/1000.f),287.f/1004.5f)/th2[h];
}

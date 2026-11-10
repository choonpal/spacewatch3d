/* Experimental CPU executor of the existing float64 BVH segment test.
 * No epsilon, triangle, sampling, policy or tree-construction changes.
 * Build without fast-math or contraction. NumPy supplies normalized rays.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
/* Ordered unequal values have an exact comparison result. Preserve libm's
 * behavior for ties (including signed zero) and unordered values verbatim. */
static inline double ordered_min(double a,double b) {
    if(a<b)return a;
    if(b<a)return b;
    return fmin(a,b);
}
static inline double ordered_max(double a,double b) {
    if(a>b)return a;
    if(b>a)return b;
    return fmax(a,b);
}
static double dot3(const double *a,const double *b) {
    return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2];
}
static void cross3(const double *a,const double *b,double *c) {
    c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];
}
int query(int64_t node_count,const double *bounds,const int64_t *nodes,
          const int64_t *indices,const double *tri,const double *origin,
          int64_t count,const double *directions,const double *ends,
          const uint8_t *active,uint8_t *out) {
    int64_t *stack=malloc((size_t)node_count*sizeof(int64_t));
    if(!stack)return 1;
    for(int64_t r=0;r<count;r++) {
        out[r]=0;if(!active[r])continue;
        const double *dir=directions+3*r;int64_t top=0;stack[top++]=0;int uncertain=0;
        while(top&&!out[r]) {
            const int64_t n=stack[--top];const double *box=bounds+6*n;
            double near=1e-7,far=ends[r];
            for(int k=0;k<3;k++) {
                if(fabs(dir[k])<1e-14) {
                    if(origin[k]<box[k]-1e-10||origin[k]>box[k+3]+1e-10)far=-1;
                } else {
                    double a=(box[k]-origin[k])/dir[k],b=(box[k+3]-origin[k])/dir[k];
                    near=ordered_max(near,ordered_min(a,b));far=ordered_min(far,ordered_max(a,b));
                }
            }
            if(fabs(near-far)<1e-12*(1+fabs(near)+fabs(far)))uncertain=1;
            if(near>far)continue;
            const int64_t *node=nodes+4*n;
            if(node[0]>=0){stack[top++]=node[0];stack[top++]=node[1];continue;}
            for(int64_t j=node[2];j<node[2]+node[3];j++) {
                const double *v=tri+9*indices[j];double e1[3],e2[3],s[3],q[3],h[3];
                for(int k=0;k<3;k++){e1[k]=v[k+3]-v[k];e2[k]=v[k+6]-v[k];s[k]=origin[k]-v[k];}
                cross3(s,e1,q);cross3(dir,e2,h);
                const double det=dot3(e1,h);
                if(fabs(fabs(det)-1e-12)<1e-14)uncertain=1;
                if(fabs(det)<=1e-12)continue;
                const double inv=1.0/det;
                const double u=inv*dot3(s,h),w=inv*dot3(q,dir),t=inv*dot3(e2,q);
                if(fabs(u+1e-10)<1e-12 || fabs(w+1e-10)<1e-12 ||
                   fabs(u+w-(1+1e-10))<1e-12 ||
                   fabs(t-1e-7)<1e-12*(1+fabs(t)) ||
                   fabs(t-ends[r])<1e-12*(1+fabs(t)+fabs(ends[r])))uncertain=1;
                if(u>=-1e-10&&w>=-1e-10&&u+w<=1+1e-10&&t>1e-7&&t<ends[r]){out[r]=1;break;}
            }
        }
        if(uncertain)out[r]=2; /* Re-run boundary cases with NumPy, not a new hit tolerance. */
    }
    free(stack);return 0;
}

"""Solve the match camera from the court lines in one frame of match.mp4.

  uv run --with opencv-python-headless --with numpy tools/calibrate_camera.py match.mp4 900

Fits the near-half court lines (baselines, sidelines, kitchen, centre) to white
pixels around hand-picked rough guides, intersects them, and runs solvePnP over
a focal-length sweep. Prints the camera in court coordinates (see index.html)
and writes calib_overlay.jpg for a visual check. The rough guides below are in
960x540 image coordinates for the fixed camera used in match.mp4.
"""
import cv2, numpy as np, json, sys
video, at = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 900
cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
ok, img = cap.read(); assert ok, "could not read frame"
H, W = img.shape[:2]
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
white=((hsv[...,1]<60)&(hsv[...,2]>170)).astype(np.uint8)
# rough lines in 960-scale (x2 for full res)
rough={
 'base':  [(20,268),(488,528)],
 'lside': [(20,260),(372,189)],
 'rside': [(492,528),(800,276)],
 'kitch': [(372,189),(800,276)],
 'center':[(165,348),(530,222)],
}
ys,xs=np.nonzero(white); P=np.stack([xs,ys],1).astype(float)
fit={}
for k,(a,b) in rough.items():
    a=np.array(a)*2.0; b=np.array(b)*2.0
    d=(b-a)/np.linalg.norm(b-a); n=np.array([-d[1],d[0]])
    t=(P-a)@d; dist=(P-a)@n
    m=(np.abs(dist)<10)&(t>40)&(t<np.linalg.norm(b-a)-40)
    Q=P[m]
    # total least squares
    c=Q.mean(0); u,s,vt=np.linalg.svd(Q-c); dd=vt[0]; nn=np.array([-dd[1],dd[0]])
    fit[k]=(nn, nn@c); print(k, m.sum())
def X(a,b):
    A=np.array([fit[a][0],fit[b][0]]); return np.linalg.solve(A,[fit[a][1],fit[b][1]])
L=6.705; K=2.134; Wd=3.048
pts={'NL':(X('base','lside'),(-L,Wd,0)),'NR':(X('base','rside'),(-L,-Wd,0)),
     'KL':(X('kitch','lside'),(-K,Wd,0)),'KR':(X('kitch','rside'),(-K,-Wd,0)),
     'CB':(X('base','center'),(-L,0,0)),'CK':(X('kitch','center'),(-K,0,0))}
for k,v in pts.items(): print(k, v[0].round(1))
img2=np.array([v[0] for v in pts.values()],np.float32); obj=np.array([v[1] for v in pts.values()],np.float32)
best=None
for f in np.linspace(800,3000,221):
    Kc=np.array([[f,0,W/2],[0,f,H/2],[0,0,1]])
    ok,r,t=cv2.solvePnP(obj,img2,Kc,None,flags=cv2.SOLVEPNP_ITERATIVE)
    pr,_=cv2.projectPoints(obj,r,t,Kc,None); e=np.sqrt(((pr[:,0]-img2)**2).sum(1)).mean()
    if best is None or e<best[0]: best=(e,f,r,t)
e,f,r,t=best; print('err',e,'f',f)
Kc=np.array([[f,0,W/2],[0,f,H/2],[0,0,1]])
R,_=cv2.Rodrigues(r); C=(-R.T@t).ravel(); fwd=R.T@np.array([0,0,1.]); up=-(R.T@np.array([0,1.,0]))
print('cam pos',C.round(3),'fwd',fwd.round(3),'up',up.round(3))
vfov=2*np.degrees(np.arctan(H/2/f)); print('vfov',vfov)
# check net posts
for name,p in [('postL',(0,3.35,0.914)),('postR',(0,-3.35,0.914)),('netC',(0,0,0.864))]:
    pr,_=cv2.projectPoints(np.array([p],np.float32),r,t,Kc,None); print(name,(pr.ravel()/2).round(1),'(960 scale)')
# overlay
ov=img.copy()
def proj(p): q,_=cv2.projectPoints(np.array([p],np.float32),r,t,Kc,None); return tuple(int(v) for v in q.ravel())
segs=[((-L,Wd),(L,Wd)),((-L,-Wd),(L,-Wd)),((-L,Wd),(-L,-Wd)),((L,Wd),(L,-Wd)),((-K,Wd),(-K,-Wd)),((K,Wd),(K,-Wd)),((-L,0),(-K,0)),((K,0),(L,0))]
for a,b in segs: cv2.line(ov,proj((*a,0)),proj((*b,0)),(0,0,255),3)
cv2.line(ov,proj((0,3.35,0.914)),proj((0,0,0.864)),(0,255,255),3); cv2.line(ov,proj((0,0,0.864)),proj((0,-3.35,0.914)),(0,255,255),3)
cv2.imwrite("calib_overlay.jpg", cv2.resize(ov, (960, 540)))
look = C + fwd * (C[2] / -fwd[2])   # where the optical axis meets the floor
print(json.dumps({"position": C.round(4).tolist(), "look_at": look.round(4).tolist(), "vfov_deg": round(vfov, 3)}))

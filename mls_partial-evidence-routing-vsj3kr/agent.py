"""Runnable starter: cheap independent-source evidence with delayed calibration."""
import math
import numpy as np

class Agent:
    def start(self,config):
        self.config=config;self.mean=np.zeros((2,8));self.var=np.ones((2,8));self.offset=np.zeros(8)
        rows=config['history'];self.prior=sum(r['label'] for r in rows)/len(rows)
        for y in (0,1):
            for s in range(8):
                values=[r['reports'][s] for r in rows if r['label']==y and r['reports'][s] is not None]
                self.mean[y,s]=np.mean(values);self.var[y,s]=max(.1,float(np.var(values)))
        return {}

    def act(self,o):
        if o['feedback'] is not None:
            f=o['feedback'];y=f['label']
            for s,v in enumerate(f['reports']):
                if v is not None:self.offset[s]+=.1*(v-self.mean[y,s]-self.offset[s])
        known=[i for i,v in enumerate(o['reports']) if v is not None]
        if len(known)<3:
            candidates=[i for i in range(8) if i not in known]
            s=max(candidates,key=lambda i:(self.mean[1,i]-self.mean[0,i])**2/((self.var[0,i]+self.var[1,i])*o['costs'][i]))
            return {'type':'query','source':int(s)}
        ll=math.log(self.prior/(1-self.prior))
        for s in known:
            z=o['reports'][s]-self.offset[s]
            ll+=-.5*(math.log(self.var[1,s]/self.var[0,s])+(z-self.mean[1,s])**2/self.var[1,s]-(z-self.mean[0,s])**2/self.var[0,s])
        p=1/(1+math.exp(-max(-40,min(40,ll))))
        audit=bool(o['case_id']%10==9 and o['audit_remaining']>0)
        if min(p,1-p)>.18:return {'type':'defer','audit':audit}
        return {'type':'decide','decision':int(p>.5),'probability':p,'audit':audit}

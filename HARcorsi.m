clear%forecast has lag of k+1 due to fact that white noise can not be known,extreme variations can not be explained by factors of the model but only by white noise
close

load('RVlog.mat');
N=length(RVlog);
K=5;

%estimating coefficient
M=realized_volatility_matrix(RVlog);
[rM,cM]=size(M);
[Beta_Hat,y_Hat,std_hat,stdBeta_Hat,betaInt,tstat,~,AdjR2,Fstat,fcrit]=OLS(rM,cM,M,RVlog(23:N));

%using all available data known at time of K step ahead forecast:we include
% new observations as we go foreward in time  
%
B_H=NaN(4,200);
std_h=NaN(200,1);
stdB_H=NaN(4,200);
for iM=1:200
[B_H(:,iM),y_Hat,std_h(iM),stdB_H(:,iM),~,~,~,~,~,~]=OLS(rM-200-K+iM,cM,M(1:rM-200-K+iM,1:cM),RVlog(23:N-200-K+iM));
end
%forecast
nm=22;
RVm=NaN(nm,200);%22*200-matrix needed for forecast function
for i_rv=1:nm
    for j_rv=1:200            %4144(N+1-200)-K=>4139
        RVm(i_rv,j_rv)=RVlog((N-200-K-nm)+ i_rv+ j_rv-1);
    end
end
forecast_K=NaN(200,1);
for jf=1:200
    forecast_K(jf)=RV_forecast_K(B_H(:,jf),RVm(:,jf),K);
end

RVreal=RVlog(N+1-200:N);
ey=  RVreal-forecast_K;
ey2=ey.^(2);
nom=sum(ey2);
RMSE=sqrt(nom/200);%residual mean square error


figure('Name','Realized volatility ')

hold on
plot(1:200,RVlog(N+1-200:N),'b')
plot(1:200,forecast_K(1:200),'r')
hold off
title('returns on log scale ');
xlabel('time');
ylabel(' log returns')
legend('true RV','Forecast RV k=5')
grid on;grid minor;



% Display the results
disp('   ');
disp('Variable      estimated     t-stat     confidence interval');
disp(['   Beta1   ''      ', num2str(Beta_Hat(1)), '      ', num2str(tstat(1)),'       ', num2str(betaInt(1,1)),' - ', num2str(betaInt(1,2)) ]);
disp(['   day     ''      ', num2str(Beta_Hat(2)), '      ', num2str(tstat(2)),'       ', num2str(betaInt(2,1)),' - ', num2str(betaInt(2,2)) ]);
disp(['   week    ''      ', num2str(Beta_Hat(3)), '      ', num2str(tstat(3)),'       ', num2str(betaInt(3,1)),' - ', num2str(betaInt(3,2)) ]);
disp(['   month   ''      ', num2str(Beta_Hat(4)), '      ', num2str(tstat(4)),'       ', num2str(betaInt(4,1)),' - ', num2str(betaInt(4,2)) ]);
disp('   ');
%disp(['R^2 Regression = ', num2str(R2)]);
disp(['adjustedR^2 Regression = ', num2str(AdjR2)]);
disp('   ');
disp(['Fstat = ', num2str(Fstat)]);
disp('  ');
disp(['Fcrit = ', num2str(fcrit)]);

disp('   ');
disp(['forecast root mean square error = ', num2str(RMSE)]);

%==========================================================================
%FUNCTIONS USED:
%==========================================================================
function forecast_k=RV_forecast_K(beta_hat,rvlog,k)
% rv=vector with the last 22 observed RV,including today(day0)
bh=beta_hat(:);
nday=1;nweek=5;nmonth=22;
dayRV=[rvlog;zeros(k,1)];
n=length(dayRV);% =nmonth+k =22+5= 27;(for k=5)
dRV=NaN(k,1);
wRV=NaN(k,1);
mRV=NaN(k,1);
for in=nmonth+1:nmonth+k
    for ik=1:k%day regressor
        
        dRV(ik)=dayRV(in-nday);%day regressor;
       
        w0=0;%week regressor
        for jw=1:nweek
            w0=w0+dayRV(in-nweek+jw);
        end
        wRV(ik)=w0/nweek;
        
        m0=0;%month regressor
        for jm=1:nmonth
            m0=m0+dayRV(in-nmonth+jm);
        end
        mRV(ik)=m0/nmonth;
        
        dayRV(in)=(bh(1)+dRV(ik)*bh(2)+wRV(ik)*bh(3)+mRV(ik)*bh(4));
    end
end
forecast_k=dayRV(nmonth+k);%expected value of RV in day0+5(ex.27th day if K=5)
end
%==========================================================================
function X=realized_volatility_matrix(rv)%month,day,week regressor matrix
nday=1; nweek=5; nmonth=22;
n=length(rv);%length of daily realized volatility

monthRV=NaN(n,1);%first nmonth values are 0
weekRV=NaN(n,1);%first nweek values are 0
dayRV=NaN(n,1);

for  im=nmonth+1:n
    m0=0 ;
    for jm=1:nmonth
        m0=m0+rv(im-jm);
    end
    monthRV(im)=m0/nmonth;
end
for  iw=nweek+1:n
    w0=0 ;
    for jw=1:nweek
        w0=w0+rv(iw-jw);
    end
    weekRV(iw)=w0/nweek;
end
for  id=nday+1:n
    d0=0 ;
    for jd=1:nday
        d0=d0+rv(id-jd);
    end
    dayRV(id)=d0/nday;
end
%or eq:for prev.dayrv ,dayRV=rv(nmonth+(1-1):n+(1-1)-1)=rv(nmonth:n-1)
X=[dayRV(nmonth+1:n),weekRV(nmonth+1:n),monthRV(nmonth+1:n)];%RVof prev day,week,month,
end
%==========================================================================
function [Beta_Hat,y_Hat,std_hat,stdBeta_Hat,betaInt,tstat,R2,AdjR2,Fstat,fcrit]=OLS(n,p,x,y)
numbetas=p+1;
X=[ones(n,1),x];        %[ones(numsimulation,1),regressor matrix]
Beta_Hat = inv(X'*X)*X'*y; % OLS fit of betas
y_Hat= X*Beta_Hat;
e = y - y_Hat;
RSS = e'*e;
sigma2_hat = RSS/(n-numbetas);% n-k fit sigma2
std_hat=sqrt(sigma2_hat);
varBeta_Hat = sigma2_hat*diag(inv(X'*X));%  estimated var of fit beta
stdBeta_Hat = sqrt(varBeta_Hat);%standard deviation
%hypothesis testing
tcrit=-tinv(.025,n);
betaInt =[ Beta_Hat-tcrit.*stdBeta_Hat,Beta_Hat+tcrit.*stdBeta_Hat];
tstat =  Beta_Hat./stdBeta_Hat;
R2 = 1- var(e)/var(y);
AdjR2=1- (var(e)/(n-numbetas))/(var(y)/(n-1));
Fstat=((n-(numbetas))/(numbetas-1))*(R2/(1-R2));
fcrit=finv(0.95,n-(numbetas),(numbetas-1));

end
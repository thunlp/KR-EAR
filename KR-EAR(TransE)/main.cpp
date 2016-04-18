
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
using namespace std;


#define pi 3.1415926535897932384626433832795


#define EXP_TABLE_SIZE 1000000
vector<double> expTable;


int L1_flag=1;

//parameter

int n = 100;
int m = 100;
double rate = 0.001;
double margin = 1;
string version = "0";
double lambda = 1.0;

int nn = 100;



map<pair<int,int>, map<pair<int,int>,double> > attr_correlation;

vector<vector<double> > attr_beta,attr_beta_tmp;
double attr_beta_bias = 0, relation_bias = 7;
vector<vector<pair<int,int> > > entity_attr_val;



//uniform distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}

//normal distribution
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}




double sqr(double x)
{
    return x*x;
}
double vec_len(vector<double> &a)
{
	double res=0;
    for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	res = sqrt(res);
	return res;
}


double f(double x)
{
	return tanh(x);
}
double f_derivative(double x)
{
	return 1-sqr(tanh(x));
}

char buf[100000],buf1[100000];

//data
int relation_num, entity_num, attribute_num, val_num;
map<string,int> relation2id,entity2id, attribute2id, val2id;
vector<vector<int> > attribute_val;
map<pair<int,int>, map<int,int> > ok_rel, ok_attr;



double res;//loss function value
vector<int> fb_h,fb_l,fb_r;
vector<int> attr_h,attr_l,attr_r;
vector<vector<double> > relation_vec, entity_vec, val_vec, attribute_vec;
vector<vector<double> > relation_tmp, entity_tmp,val_tmp, attribute_vec_tmp;

vector<vector<vector<double> > > attribute_matrix, attribute_tmp;



map<int,map<int,vector<int> > > left_entity,right_entity;
map<int,double> left_mean,right_mean;

void add_rel(int x,int y,int z)
{
    fb_h.push_back(x);
    fb_r.push_back(z);
    fb_l.push_back(y);
    ok_rel[make_pair(x,z)][y]=1;
}
void add_attr(int x,int y,int z)
{
    attr_h.push_back(x);
    attr_r.push_back(z);
    attr_l.push_back(y);
    ok_attr[make_pair(x,z)][y]=1;
}
double norm(vector<double> &a, double y)
{
	while (true)
	{
	    double x = vec_len(a);
	    if (x>y)
		{
			res += lambda*(x*x-y);
	    	for (int ii=0; ii<a.size(); ii++)
	            	a[ii]-=2*rate*lambda*a[ii];
		}
		else
	    	return 0;
	}
}

double count0, count1;
double norm(int e1, int rel)
{
	while (true)
	{
		vector<double> syn1;
		for (int j=0; j<m; j++)
		{
			double tmp = attribute_vec_tmp[rel][j];
			for (int i=0; i<n; i++)
				tmp += entity_tmp[e1][i]*attribute_tmp[rel][i][j];
			syn1.push_back(tmp);
		}
		double y = 0;
		for (int j=0; j<m; j++)
			y+= sqr(tanh(syn1[j]));
		count0 +=y;
		count1+=1;
		if (y>1)
		{
			res += 1*lambda*(y-1);
			for (int j=0; j<m; j++)
			{
				double tmp = syn1[j];
				double x = 2*tanh(tmp);
				for (int i=0; i<n; i++)
				{
					double tmp_val = entity_tmp[e1][i];
					entity_tmp[e1][i] -= 1*rate*lambda*x*f_derivative(tmp)*attribute_tmp[rel][i][j];
					attribute_tmp[rel][i][j] -= 1*rate*lambda*x*f_derivative(tmp)*tmp_val;
				}
				attribute_vec_tmp[rel][j] -= 1*rate*lambda*x*f_derivative(tmp);
			}
		}
		else
			break;
	}
}
int rand_max(int x)
{
    int res = (rand()*rand())%x;
    while (res<0)
        res+=x;
    return res;
}

double MAX_EXP = 6;
double sigmoid(double x)
{
	if (x>MAX_EXP)
		return 1;
	else
		if (x<-MAX_EXP)
			return 0;
	else
	{
		return expTable[(int)((x + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
		//return 1/(1+exp(-x));
	}
}


double rel_rate = 1;
double attr_rate = 1;

double entity_bias = 7;

void gradient(double sig, int e1,int e2,int rel, int y)
{
    for (int ii=0; ii<n; ii++)
    {

        double x = 2*(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        if (L1_flag)
        	if (x>0)
        		x=1;
        	else
        		x=-1;
        relation_tmp[rel][ii]+=rate*rel_rate*x*(y-sig);
        entity_tmp[e1][ii]+=rate*rel_rate*x*(y-sig);
        entity_tmp[e2][ii]-=rate*rel_rate*x*(y-sig);
    }
	//entity_bias += rate*(y-sig);
}


double calc_sum(int e1,int e2,int rel, int y)
{
    double sum=0;
    if (L1_flag)
    	for (int ii=0; ii<n; ii++)
        	sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
    else
    	for (int ii=0; ii<n; ii++)
        	sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
	double sig = sigmoid(rel_rate*(-sum + entity_bias));
	res += fabs(y-sig);
	gradient(sig, e1, e2, rel, y);
    return sum;
}

void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
{
    double sum1 = calc_sum(e1_a,e2_a,rel_a, 1);
    double sum2 = calc_sum(e1_b,e2_b,rel_b, 0);
	/*
    if (sum1+margin>sum2)
    {
    	res+=margin+sum1-sum2;
    	gradient( e1_a, e2_a, rel_a, rate*-1);
		gradient(e1_b, e2_b, rel_b, rate);
    }*/
}

int sample_attr_num = 1;
vector<int> sample_attr;

void gradient_attr(double sig, double sig1,int e1,int val,int attr, int y)
{
	//rate *=10;
	for (int j=0; j<m; j++)
	{
		double tmp = attribute_vec[attr][j];
		for (int i=0; i<n; i++)
			tmp += entity_vec[e1][i]*attribute_matrix[attr][i][j];
		double x = val_vec[val][j];
		if (val_vec[val][j] - f(tmp)> 0)
			x = 1;
		else
			x = -1;
		val_tmp[val][j] -= attr_rate*rate * x * (y- sig);
		for (int i=0; i<n; i++)
		{
			entity_tmp[e1][i] += attr_rate*rate*x*f_derivative(tmp)*attribute_matrix[attr][i][j]* (y- sig);
			attribute_tmp[attr][i][j] += attr_rate*rate*x*f_derivative(tmp)*entity_vec[e1][i]* (y- sig);
		}
		attribute_vec_tmp[attr][j] += attr_rate*rate*x*f_derivative(tmp)* (y- sig);
	/*	double x = entity_vec[e1][j] + attribute_vec[rel][j];
		if (val_vec[val][j] - x> 0)
			x = 1;
		else
			x = -1;
		val_tmp[val][j] -= rate * x * (y- sigmoid(sum+7));
		entity_tmp[e1][j] += rate * x * (y- sigmoid(sum+7));
		attribute_vec_tmp[rel][j] += rate*x* (y- sigmoid(sum+7));*/
		
	}
//	relation_bias += rate*(y-sig);
	int id_num = entity_attr_val[e1].size();
	for (int id=0; id<id_num; id++)
	{
		int k = sample_attr[id];
		int attr1 = entity_attr_val[e1][k].first;
		int val1 = entity_attr_val[e1][k].second;
	//	double pr = attr_correlation[make_pair(attr1,val1)][make_pair(attr,val)];
		if (attr_correlation[make_pair(attr1,val1)].count(make_pair(attr,val)) > 0)
		{
			double delta = 1.0/id_num*rate* attr_correlation[make_pair(attr1,val1)][make_pair(attr,val)]*(y- sig1);
			for (int i=0; i<nn; i++)
			{
				attr_beta_tmp[attr][i] += delta*attr_beta[attr1][i];
				attr_beta_tmp[attr1][i] += delta*attr_beta[attr][i];
			}
		}
	}
	//attr_beta_bias += rate*(y-sig1);
	//rate *=0.1;
}

double dis_attr(int attr, int val, int attr1, int val1)
{
    double sum=0;
	for (int j=0; j<m; j++)
	{
		double tmp = 0;
		for (int i=0; i<n; i++)
			tmp += (val_vec[val][i]-attribute_vec[attr][i])*attribute_matrix[attr1][i][j];
		for (int i=0; i<n; i++)
			tmp -= (val_vec[val1][i]-attribute_vec[attr1][i])*attribute_matrix[attr][i][j];
		sum += fabs(tmp);
	}
	return sum;
}



double attr_sim(int attr, int attr1)
{
	double res = 0;
	for (int i=0; i<nn; i++)
		res+= attr_beta[attr][i]*attr_beta[attr1][i];
	return res;
}

double res1 = 0;
double calc_attr(int e1, int val, int attr, int y)
{
    double sum=0;
    vector<double> syn1;
	for (int j=0; j<m; j++)
	{
		double tmp = attribute_vec[attr][j];
		for (int i=0; i<n; i++)
			tmp += entity_vec[e1][i]*attribute_matrix[attr][i][j];
	//	double tmp =  entity_vec[e1][j] + attribute_vec[rel][j];
		syn1.push_back(f(tmp));
		//syn1.push_back(tmp);
	}
	for (int ii=0; ii<m; ii++)
		sum+=-fabs(val_vec[val][ii] - syn1[ii]);
	double sum1 = 0;
	int id_num = entity_attr_val[e1].size();
	sample_attr.clear();
	if (id_num>0)
	for (int id=0; id<id_num; id++)
	{
		int k = rand_max(id_num);
		sample_attr.push_back(k);
		int attr1 = entity_attr_val[e1][k].first;
		int val1 = entity_attr_val[e1][k].second;
		if (attr_correlation[make_pair(attr1,val1)].count(make_pair(attr,val)) > 0)
			sum1 += attr_correlation[make_pair(attr1,val1)][make_pair(attr,val)]/id_num*attr_sim(attr,attr1);
			//sum1-=dis_attr(attr,val,attr1,val1)/sample_attr_num;
	}
	/*if (id_num ==0)
		//return 0;
		attr_rate = 1;
	else
		if (rand_max(2) ==1)
			attr_rate = 1;
		else
			attr_rate = 1;// 0.0;*/
	//cout<<sum<<' '<<sum1<<' '<<y<<endl;
	//attr_rate=0.5;
	double sig = sigmoid(sum+7);//relation_bias);//sigmoid((attr_rate*sum + (1-attr_rate)*sum1 + 7));
	double sig1 = sigmoid(sum1-2);//attr_beta_bias);
	res += fabs(y-sig);
	res += fabs(y-sig1);
	res1 += 1;
	gradient_attr(sig, sig1, e1, val, attr, y);
    return sum;
}



void train_attr(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
{
	//cout<<"205"<<endl;
    double sum1 = calc_attr(e1_a,e2_a,rel_a, 1);
    double sum2 = calc_attr(e1_b,e2_b,rel_b, 0);
	//cout<<sum1<<' '<<sigmoid(sum1 + 7)<<' '<<sum2<<' '<<sigmoid(sum2 + 7)<<endl;
}

void sgd()
{
//	cout<<220<<endl;
    res=0;
    int nbatches=100;
    int nepoch = 1000;
    int batchsize = fb_h.size()/nbatches;
	int batchsize2 = attr_h.size()/nbatches;
	//cout<<batchsize<<' '<<batchsize2<<endl;
    for (int epoch=0; epoch<nepoch; epoch++)
    {

    	res=0;
     	for (int batch = 0; batch<nbatches; batch++)
     	{
     		for (int k=0; k<batchsize; k++)
     		{
				int i=rand_max(fb_h.size());
				int e1 = fb_h[i],rel = fb_r[i], e2  = fb_l[i];
				int j=0;
				calc_sum(e1,e2,rel, 1);
				for (int neg_id = 0; neg_id<10; neg_id++)
				{

					double pr = 500;// 1000*right_mean[fb_r[i]]/(right_mean[fb_r[i]]+left_mean[fb_r[i]]);
					j = rand_max(entity_num);
					pr = rand()%1000;
					if (pr<333)//rand()%1000<pr)
					{
						while (ok_rel[make_pair(e1,rel)].count(j)>0)
							j=rand_max(entity_num);
						calc_sum(e1,j,rel, 0);
					}
					else
					if (pr<666)
					{
						while (ok_rel[make_pair(j,rel)].count(e2)>0)
							j=rand_max(entity_num);
						calc_sum(j,e2,rel, 0);

					}
					else
					{
					int rel_neg = rand_max(relation_num);
					while (ok_rel[make_pair(e1,rel_neg)].count(e2)>0)
						rel_neg = rand_max(relation_num);
					calc_sum(e1,e2,rel_neg, 0);
					norm(relation_tmp[rel_neg],1);
					}
	        		norm(entity_tmp[j],1);
				}
				norm(relation_tmp[rel],1);
        		norm(entity_tmp[e1],1);
        		norm(entity_tmp[e2],1);
     		}
     		for (int k=0; k<batchsize2/10; k++)
     		{
				int i=rand_max(attr_h.size());
				int e1 = attr_h[i],attr = attr_r[i], val  = attr_l[i];
				calc_attr(e1,val,attr, 1);
				for (int neg_id = 0; neg_id<1; neg_id++)
				{
					double pr = 1000*right_mean[attr]/(right_mean[attr]+left_mean[attr]);
					if (rand()%1000<1000)
					{
						int j=attribute_val[attr][rand_max(attribute_val[attr].size())];
						int times = 0;
						while (ok_attr[make_pair(e1,attr)].count(j)>0)
						{
							j=attribute_val[attr][rand_max(attribute_val[attr].size())];
							times++;
							if (times>3)
								break;
						}
						if (times>3)
							continue;
					   // train_attr(e1,val,rel,e1,j,rel);
						calc_attr(e1,j,attr, 0);
						norm(val_tmp[j],1);
					}
					else
					{
						int j = rand_max(entity_num);
						while (ok_attr[make_pair(j,attr)].count(val)>0)
							j = rand_max(entity_num);
					 //   train_attr(e1,val,rel,j,val,rel);
						calc_attr(j,val,attr, 0);
						norm(entity_tmp[j],1);
					}
				}

					norm(entity_tmp[e1],1);
					norm(val_tmp[val],1);
					//norm(attribute_vec[rel],0);
					norm(e1,attr);
     		}
			//for (int i=0; i<attribute_num; i++)
			//	norm(attr_beta_tmp[i],1);
            relation_vec = relation_tmp;
            entity_vec = entity_tmp;
			val_vec = val_tmp;
			attribute_matrix = attribute_tmp;
			attribute_vec = attribute_vec_tmp;
			attr_beta = attr_beta_tmp;
     	}
		for (int i=0; i<10; i++)
			cout<<vec_len(attr_beta[i])<<' ';

		cout<<endl;
        cout<<"epoch:"<<epoch<<' '<<res<<' '<<relation_bias<<' '<<attr_beta_bias<<' '<<entity_bias<<endl;
		res1 = 0;
        FILE* f2 = fopen(("relation2vec"+version+".txt").c_str(),"w");
        FILE* f3 = fopen(("entity2vec"+version+".txt").c_str(),"w");
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
            fprintf(f2,"\n");
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
            fprintf(f3,"\n");
        }
        fclose(f2);
        fclose(f3);
		FILE* f_val = fopen(("val2vec"+version+".txt").c_str(),"w");
		for (int i=0; i<val_num; i++)
		{
			for (int ii=0; ii<m; ii++)
				fprintf(f_val, "%.6lf\t", val_vec[i][ii]);
			fprintf(f_val,"\n");
		}
		fclose(f_val);
		FILE* f_attr = fopen(("attr2matrix"+version+".txt").c_str(),"w");
		for (int i=0; i<attribute_num; i++)
		{
			for (int ii=0; ii<n; ii++)
			{
				for (int jj=0; jj<m; jj++)
					fprintf(f_attr, "%.6lf\t", attribute_matrix[i][ii][jj]);
				fprintf(f_attr,"\n");
			}
		}
		fclose(f_attr);
		FILE* f_attr1 = fopen(("attr2vec"+version+".txt").c_str(),"w");
		for (int i=0; i<attribute_num; i++)
		{
				for (int jj=0; jj<m; jj++)
					fprintf(f_attr1, "%.6lf\t", attribute_vec[i][jj]);
				fprintf(f_attr1,"\n");
		}
		fclose(f_attr1);
		FILE* f_attr2 = fopen(("attr2beta"+version+".txt").c_str(),"w");
		for (int i=0; i<attribute_num; i++)
		{
				for (int jj=0; jj<nn; jj++)
					fprintf(f_attr2, "%.6lf\t", attr_beta[i][jj]);
				fprintf(f_attr2,"\n");
		}
		fclose(f_attr2);
    }
}


void run()
{
    relation_vec.resize(relation_num);
	for (int i=0; i<relation_vec.size(); i++)
		relation_vec[i].resize(n);
    entity_vec.resize(entity_num);
	for (int i=0; i<entity_vec.size(); i++)
		entity_vec[i].resize(n);
    attribute_matrix.resize(attribute_num);
	for (int i=0; i<attribute_matrix.size(); i++)
	{
		
		attribute_matrix[i].resize(n);
		for (int j= 0; j<n; j++)
			attribute_matrix[i][j].resize(m);
	}
    val_vec.resize(val_num);
	for (int i=0; i<val_vec.size(); i++)
		val_vec[i].resize(m);
    attribute_vec.resize(attribute_num);
	for (int i=0; i<attribute_vec.size(); i++)
		attribute_vec[i].resize(m);
	
    for (int i=0; i<relation_num; i++)
    {
        for (int ii=0; ii<n; ii++)
            relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
    }
    for (int i=0; i<entity_num; i++)
    {
        for (int ii=0; ii<n; ii++)
            entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        norm(entity_vec[i],1);
    }
//	cout<<382<<endl;
	for (int i=0; i<attribute_matrix.size(); i++)
	{
		for (int j= 0; j<n; j++)
			for (int k=0; k<m; k++)
				if (j==k)
				attribute_matrix[i][j][k] = 1;//rand(-6/sqrt(m),6/sqrt(m));
		else
			attribute_matrix[i][j][k] = 0;
	}
//	cout<<391<<endl;
    for (int i=0; i<val_num; i++)
    {
        for (int ii=0; ii<m; ii++)
            val_vec[i][ii] = randn(0,1.0/m,-6/sqrt(m),6/sqrt(m));
        norm(val_vec[i],1);
    }
//	cout<<397<<endl;
    for (int i=0; i<attribute_num; i++)
    {
        for (int ii=0; ii<m; ii++)
            attribute_vec[i][ii] = randn(0,1.0/m,-6/sqrt(m),6/sqrt(m));
       // norm(attribute_vec[i]);
    }
	//cout<<"here"<<endl;
	attr_beta.resize(attribute_num);
	for (int i=0; i<attribute_num; i++)
	{
		attr_beta[i].resize(nn);
		for (int ii=0; ii<nn; ii++) 
			attr_beta[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
	}
//	cout<<403<<endl;
	cout<<"init finish"<<endl;
	relation_tmp=relation_vec;
	entity_tmp = entity_vec;
	attribute_tmp = attribute_matrix;
	val_tmp = val_vec;
	attribute_vec_tmp = attribute_vec;
	attr_beta_tmp = attr_beta;
	sgd();
}

void prepare()
{
	int x;
	FILE* f_rel = fopen("../data/relation2id.txt","r");
	while (fscanf(f_rel,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		relation_num++;
	}
	fclose(f_rel);
	FILE* f_attr = fopen("../data/attribute2id.txt","r");
	while (fscanf(f_attr,"%s%d",buf,&x)==2)
	{
		string st=buf;
		attribute2id[st]=x;
		attribute_num++;
	}
	fclose(f_attr);
	FILE* f_env = fopen("../data/entity2id.txt","r");
	while (fscanf(f_env,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		entity_num++;
	}
	fclose(f_env);
	FILE* f_val = fopen("../data/val2id.txt","r");
	while (fscanf(f_env,"%s%d",buf,&x)==2)
	{
		string st=buf;
		val2id[st]=x;
		val_num++;
	}
	fclose(f_env);
	attribute_val.resize(relation_num);
	entity_attr_val.resize(entity_num);
	cout<<"here"<<endl;
	FILE* f3 = fopen("../data/attribute_val.txt", "r");
	while (fscanf(f3,"%s%d",buf,&x)==2)
	{
		string s1 = buf;
		int attr = -1;
		if (attribute2id.count(s1)>0)
			attr = attribute2id[s1];
		for (int i=0; i<x; i++)
		{
			fscanf(f3,"%s",buf);
			string s2 = buf;
			if (attr>=0)
			{
				if (val2id.count(s2)==0)
					continue;
				attribute_val[attr].push_back(val2id[s2]);
			}
		}
	}
	cout<<"here"<<endl;
    FILE* f_kb = fopen("../data/train-rel.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
			cout<<"miss entity"<<s1<<endl;
        if (relation2id.count(s3)>0)
        {
	        
	        if (entity2id.count(s2)==0)
	        	cout<<"miss entity"<<s1<<endl;
            add_rel(entity2id[s1],entity2id[s2],relation2id[s3]);
	       /* left_entity[relation2id[s3]][entity2id[s1]].push_back(entity2id[s2]);
	        right_entity[relation2id[s3]][entity2id[s2]].push_back(entity2id[s1]);*/
        }
		else
		if (attribute2id.count(s3)>0)
		{
			cout<<"?"<<endl;
			//cout<<"here"<<endl;
			//add_attr(entity2id[s1],entity2id[s2],attribute2id[s3]);
		}
		else
			cout<<"miss rel/attr:"<<s3<<endl;
    }
	fclose(f_kb);
    FILE* f_kb1 = fopen("../data/train-attr.txt","r");
	while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
			cout<<"miss entity"<<s1<<endl;
        if (relation2id.count(s3)>0)
        {

			cout<<"369\t"<<s3<<' '<<relation2id[s3]<<endl;
	        
        }
		else
		if (attribute2id.count(s3)>0)
		{
	        if (val2id.count(s2)==0)
	        	cout<<"miss val"<<s1<<endl;
			int e1 = entity2id[s1];
			int attr = attribute2id[s3];
			int val = val2id[s2];
            add_attr(entity2id[s1],val2id[s2],attribute2id[s3]);
			entity_attr_val[e1].push_back(make_pair(attr,val));
			left_entity[attr][e1].push_back(val);
			right_entity[attr][val].push_back(e1);
		}
		else
			cout<<"miss rel/attr:"<<s3<<endl;
    }
	fclose(f_kb1);
	
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0,sum3 = 0;
    	for (map<int,vector<int> >::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second.size();
    	}
    	left_mean[i]=sum2/sum1;

    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0,sum3=0;
    	for (map<int,vector<int> >::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second.size();
    	}
    	right_mean[i]=sum2/sum1;
    }
	
	FILE* f_attr_correlation = fopen("limit/attr_correlation.txt", "r");
	while ((fscanf(f_attr_correlation,"%s",buf)==1))
	{
        string s1=buf;
        fscanf(f_attr_correlation,"%s",buf);
        string s2=buf;
        fscanf(f_attr_correlation,"%s",buf);
        string s3=buf;
        fscanf(f_attr_correlation,"%s",buf);
        string s4=buf;
		double pr;
		fscanf(f_attr_correlation, "%lf", &pr);
		int attr = attribute2id[s1], val = val2id[s2];
		int attr1 = attribute2id[s3], val1 = val2id[s4];
		attr_correlation[make_pair(attr,val)][make_pair(attr1,val1)] = pr;
	}
	
	cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
	cout<<"attribute_num="<<attribute_num<<endl;
	cout<<"val_num="<<val_num<<endl;
	cout<<fb_h.size()<<" "<<attr_h.size()<<endl;
   // fclose(f_kb);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    int i;
    if ((i = ArgPos((char *)"-n", argc, argv)) > 0) n = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-m", argc, argv)) > 0) m = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-version", argc, argv)) > 0) version = argv[i + 1];
	
    cout<<"n = "<<n<<endl;
	cout<<"m = "<<m<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;
	cout<<"version = "<<version<<endl;
	for (i = 0; i < EXP_TABLE_SIZE; i++) 
	{
		double tmp = exp((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable.push_back(tmp/(tmp+1));                   // Precompute f(x) = x / (x + 1)
	}
    prepare();
	run();
}



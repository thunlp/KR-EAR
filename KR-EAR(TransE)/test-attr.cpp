#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
using namespace std;

bool debug=false;
int L1_flag=1;

int nn=100;

string version;
string trainortest = "test";

map<string,int> attribute2id,entity2id, val2id, relation2id;

vector<vector<double> > attr_beta;


map<int, vector<int> > attribute_val;

map<int,map<int,int> > entity_rel_left, entity_rel_right;

map<int, vector<pair<int,int> > > link_left, link_right;

struct attr_val{
	int attr;
	int val;
	double pr;
	attr_val(int x, int y, double f)
	{
		attr = x;
		val = y;
		pr = f;
	}
};

vector<vector<attr_val> > limit_left, limit_right;

map<int, map< pair<int,int>, vector<attr_val> > > left2right, right2left;

map<pair<int,int>, map<pair<int,int>,double> > attr_correlation;
vector<vector<pair<int,int> > > entity_attr_val;


int attribute_num,entity_num, val_num, relation_num;
int n = 100;
int m = 100;

double sigmoid(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%10==9)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}


map<int,map<int,double> > cover;
int cover_num = 0, cover1_num = 0;
int step;

class Test{
    vector<vector<double> > attribute_vec,entity_vec, val_vec;

	vector<vector<vector<double> > > attribute_matrix;


    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    void add(int x,int y,int z, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
	double calc_attr_origin(int e1, int val, int attr)
	{
	    double sum=0;
	    vector<double> syn1;
		for (int j=0; j<m; j++)
		{
			double tmp = attribute_vec[attr][j];
			for (int i=0; i<n; i++)
				tmp += entity_vec[e1][i]*attribute_matrix[attr][i][j];
			syn1.push_back(tanh(tmp));
		}
		for (int ii=0; ii<m; ii++)
		    if (L1_flag==1)
		        sum+=-fabs(val_vec[val][ii]-syn1[ii]);
		    else
				if (L1_flag==2)
					sum=val_vec[val][ii]*syn1[ii];
			else
		    	sum+=-sqr(val_vec[val][ii]-syn1[ii]);
		//if (L1_flag==2)
	//	sum = 1/(1+exp(-(sum+7)));
		return sum;
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
	
	double calc_attr(int e1, int val, int attr)
	{
		double sum = calc_attr_origin(e1,val,attr);
		double sum1 = 0;

	/*	int id_num = entity_attr_val[e1].size();
		for (int id=0; id<id_num; id++)
		{
			int k = id;
			int attr1 = entity_attr_val[e1][k].first;
			int val1 = entity_attr_val[e1][k].second;
			//double pr = attr_correlation[make_pair(attr1,val1)][make_pair(attr,val)];
			if (attr_correlation[make_pair(attr1,val1)].count(make_pair(attr,val)) > 0)
				sum1+= attr_correlation[make_pair(attr1,val1)][make_pair(attr,val)]/id_num  *attr_sim(attr,attr1);
			//sum1-=dis_attr(attr,val,attr1,val1)/id_num;
		}
			//cout<<sum<<' '<<sum1<<' '<<ok[make_pair(e1,attr)].count(val)<<endl;
		sum = sigmoid(sum + 7)* sigmoid(sum1 - 2);*/
		/*
	//	cout<<e1<<' '<<val<<' '<<rel<<endl;
	   
		//	cout<<L1_flag<<' '<<vec_len(syn1)<<' '<<vec_len(val_vec[val])<<' '<<sum<<endl;
		double sum1 = 0;
		double sum2 = 0;
		for (map<int,int>::iterator it = entity_rel_left[e1].begin(); it!=entity_rel_left[e1].end(); it++)
		{
			int rel = it->first;
			for (int i =0; i<limit_left[rel].size(); i++)
			{
				int attr1 = limit_left[rel][i].attr;
				int val1 = limit_left[rel][i].val;
				double pr1 = limit_left[rel][i].pr;
				//pr1*=pr1;
				if (attr1 == attr)
				{
					if (val == val1)
						sum1 += pr1;
					else
						sum1 -= pr1;
					sum2 += pr1;
				}
			}
		}
		//if (sum2>0)
		//	sum -= 1.0*sum1/sum2;
		//sum1 = 0;
		//sum2 = 0;
		for (map<int,int>::iterator it = entity_rel_right[e1].begin(); it!=entity_rel_right[e1].end(); it++)
		{
			int rel = it->first;
			for (int i =0; i<limit_right[rel].size(); i++)
			{
				int attr1 = limit_right[rel][i].attr;
				int val1 = limit_right[rel][i].val;
				double pr1 = limit_right[rel][i].pr;
				//pr1*=pr1;
				if (attr1 == attr)
				{
					if (val == val1)
						sum1 += pr1;
					else
						sum1 -= pr1;
					sum2 += pr1;
				}
			}
		}
		if (sum2>0)
		{
		//	sum += 2*sum1/sum2;
		//	cover[step][val] = 1;
		}
		double sum3 = 0;
		for (int id=0; id<entity_attr_val[e1].size(); id++)
		{
			int attr1 = entity_attr_val[e1][id].first;
			int val1 = entity_attr_val[e1][id].second;
			if (attr_correlation[make_pair(attr1,val1)].count(make_pair(attr,val)) > 0)
			{
				sum3+=attr_correlation[make_pair(attr1,val1)][make_pair(attr,val)];
				//cout<<"here"<<endl;
			}
		}
		if (entity_attr_val[e1].size()>0)
			sum3/=entity_attr_val[e1].size();
		//if (ok[make_pair(e1,attr)].count(val))
		//	cout<<sum3<<' '<<entity_attr_val[e1].size()<<endl;
			//if (sum4>0)
		//sum+= sum3;
	/*	if (sum3>0)
		{
			cover[step][val] = 1;
		}
		double sum5 = 0;
		for (int j=0; j<link_right[e1].size(); j++)
		{
			int rel = link_right[e1][j].first;
			int e2 = link_right[e1][j].second;
			if (left2right[rel].count(make_pair(attr,val))>0)
			{
				for (int i=0; i<left2right[rel][make_pair(attr,val)].size(); i++)
				{
					int attr1 = left2right[rel][make_pair(attr,val)][i].attr;
					int val1 = left2right[rel][make_pair(attr,val)][i].val;
					double pr1 = left2right[rel][make_pair(attr,val)][i].pr;
					if (entity_attr_val[e2].count(make_pair(attr1,val1))>0)
						sum5 += pr1;
				}
			}
		}*/
		//if (ok[make_pair(e1,attr)].count(val))
		//	cout<<sum<<' '<<sum5<<endl;
		//sum += sum5;*/
	    return sum;
	}
    void run()
    {
        FILE* f1 = fopen(("attr2vec"+version+".txt").c_str(),"r");
        FILE* f3 = fopen(("entity2vec"+version+".txt").c_str(),"r");
        cout<<attribute_num<<' '<<entity_num<<endl;
        attribute_vec.resize(attribute_num);
        for (int i=0; i<attribute_num;i++)
        {
            attribute_vec[i].resize(m);
            for (int ii=0; ii<m; ii++)
                fscanf(f1,"%lf",&attribute_vec[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
        }
        fclose(f1);
        fclose(f3);
	//	cout<<138<<endl;
		FILE* f_attr = fopen(("attr2matrix"+version+".txt").c_str(),"r");
		attribute_matrix.resize(attribute_num);
		for (int i=0; i<attribute_num; i++)
		{
			attribute_matrix[i].resize(n);
			for (int ii=0; ii<n; ii++)
			{
				attribute_matrix[i][ii].resize(m);
				for (int jj=0; jj<m; jj++)
					fscanf(f_attr, "%lf", &attribute_matrix[i][ii][jj]);
			}
		}
		fclose(f_attr);
	//	cout<<152<<endl;
		FILE* f_val = fopen(("val2vec"+version+".txt").c_str(),"r");
		val_vec.resize(val_num);
		for (int i=0; i<val_num; i++)
		{
			val_vec[i].resize(m);
			for (int ii=0; ii<m; ii++)
				fscanf(f_val, "%lf", &val_vec[i][ii]);
		}
		fclose(f_val);
		
		FILE* f_attr2 = fopen(("attr2beta"+version+".txt").c_str(),"r");
		attr_beta.resize(attribute_num);
		for (int i=0; i<attribute_num; i++)
		{
			attr_beta[i].resize(nn);
			for (int jj=0; jj<nn; jj++)
				fscanf(f_attr2, "%lf", &attr_beta[i][jj]);
		}
		fclose(f_attr2);
		
		
		double rsum = 0,rsum_filter=0;
		double rp_n=0,rp_n_filter = 0;
		//cout<<164<<endl;
		int hit_n = 1;

		double lll = 999999;
		double rrr = -324234234234;
		double mean =0;
		double tot_mean = 0;
		map<int, double> distribution;
		for (int testid = 0; testid<fb_l.size(); testid+=1)
		{
			step = testid;
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
			/*if (l==7)
			{
				cout<<"-------------------------"<<endl;
				cout<<l<<' '<<rel<<endl;
				cout<<attr_correlation.count(make_pair(rel,l))<<endl;
				for (int i=0; i<attr_correlation[make_pair(rel,l)].size(); i++)
				{
					int attr1 = attr_correlation[make_pair(rel,l)][i].attr;
					int val1 = attr_correlation[make_pair(rel,l)][i].val;
					cout<<entity_attr_val[h].count(make_pair(attr1,val1))<<' '<<attr1<<' '<<val1<<endl;
				}
				cout<<endl;
			}*/
			vector<pair<int,double> > a;
			for (int i=0; i<attribute_val[rel].size(); i++)
			{
			//	cout<<i<<' '<<attribute_val[rel][i]<<endl;
				double sum = calc_attr(h,attribute_val[rel][i],rel);
				/*cout<<i<<' '<<attribute_val[rel][i]<<' '<<sum<<endl;
				vec_output(val_vec[attribute_val[rel][i]]);
				vec_output(entity_vec[h]);
				vec_output(attribute_vec[rel]);*/
				a.push_back(make_pair(attribute_val[rel][i],sum));
			}
			sort(a.begin(),a.end(),cmp);
			int ttt=0;
			int filter=0;
			if (cover.count(step)> 0)
			{
				cover_num+=1;

				/*cout<<l<<"------------------"<<a[a.size()-1].first<<endl;
				for (int i=0; i<val_num; i++)
					if (cover[step].count(i)>0)
						cout<<i<<' ';
				cout<<endl;*/
				if (cover[step].count(l)>0)
				{
					cover1_num+=1;
				}
			}
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(h,rel)].count(a[i].first)>0)
				{
					mean+=a[i].second;
					lll = min(lll,a[i].second);
					rrr = max(rrr,a[i].second);
					tot_mean+=1;
					distribution[int(a[i].second)]++;
				}
				//cout<<a[i].second<<' '<<ok[make_pair(h,rel)].count(a[i].first)<<' ';
				if (ok[make_pair(h,rel)].count(a[i].first)>0)
					ttt++;
				if (ok[make_pair(h,rel)].count(a[i].first)==0)
			    	filter+=1;
				if (a[i].first==l)
				{
					rsum+=a.size()-i;
					rsum_filter+=filter+1;
					if (a.size()-i<=hit_n)
					{
						rp_n+=1;
					}
					//cout<<"filter:"<<filter<<' '<<a.size()-i<<' '<<ok[make_pair(h,rel)].size()<<' '<<rel<<endl;
					
					if (filter<hit_n)
					{
						rp_n_filter+=1;
					}
					break;
				}
			}
		//	cout<<endl;
			if (testid%100==0)
			{
				cout<<testid<<":"<<endl;
				cout<<"right:"<<rsum/(testid+1)<<'\t'<<rp_n/(testid+1)<<'\t'<<rsum_filter/(testid+1)<<'\t'<<rp_n_filter/(testid+1)<<endl;
			}
		}
		cout<<cover.size()<<' '<<cover_num<<' '<<cover1_num<<endl;
		cout<<"right:"<<rsum/fb_r.size()<<'\t'<<rp_n/fb_r.size()<<'\t'<<rsum_filter/fb_r.size()<<'\t'<<rp_n_filter/fb_r.size()<<endl;
		cout<<lll<<' '<<rrr<<' '<<mean/tot_mean<<endl;
		for (int i=-20; i<20; i++)
			cout<<i<<": "<<distribution[i]<<endl;;
    }

};
Test test;

void prepare()
{
    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/attribute2id.txt","r");
	FILE* f3 = fopen("../data/val2id.txt","r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		entity_num++;
	}
	//cout<<225<<endl;
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		attribute2id[st]=x;
		attribute_num++;
	}
	//cout<<231<<endl;
	while (fscanf(f3,"%s%d",buf,&x)==2)
	{
		string st=buf;
		val2id[st]=x;
		val_num++;
	}
	fclose(f1);
	fclose(f2);
	fclose(f3);
	FILE* f5 = fopen("../data/relation2id.txt", "r");
	while (fscanf(f5,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		relation_num++;
	}
	fclose(f5);
	//cout<<237<<endl;
	entity_attr_val.resize(entity_num);
	FILE* f4 = fopen("../data/attribute_val.txt", "r");
	while (fscanf(f4,"%s%d",buf,&x)==2)
	{
	//	cout<<string(buf)<<' '<<x<<endl;
		string s1 = buf;
		int attr = -1;
		if (attribute2id.count(s1)>0)
			attr = attribute2id[s1];
		for (int i=0; i<x; i++)
		{
			fscanf(f4,"%s",buf);
			string s2 = buf;
			if (attr>=0)
			{
				if (val2id.count(s2)==0)
					continue;
				attribute_val[attr].push_back(val2id[s2]);
			}
		}
	}
	fclose(f4);
	cout<<entity_num<<' '<<val_num<<' '<<attribute_num<<endl;
    FILE* f_kb = fopen("../data/test-attr.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (val2id.count(s2)==0)
        {
            cout<<"miss val:"<<s2<<endl;
        }
        if (attribute2id.count(s3)==0)
        {
        	cout<<"miss attribute:"<<s3<<endl;
        }
        test.add(entity2id[s1],val2id[s2],attribute2id[s3],true);
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
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (val2id.count(s2)==0)
        {
            cout<<"miss val:"<<s2<<endl;
        }
        if (attribute2id.count(s3)==0)
        {
            cout<<"miss attribute:"<<s3<<endl;
        }
		int e1 = entity2id[s1];
		int attr = attribute2id[s3];
		int val = val2id[s2];
        test.add(entity2id[s1],val2id[s2],attribute2id[s3],false);
		entity_attr_val[e1].push_back(make_pair(attr,val));
    }
    fclose(f_kb1);
	
    FILE* f_rel = fopen("../data/train-rel.txt","r");
	while (fscanf(f_rel,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_rel,"%s",buf);
        string s2=buf;
        fscanf(f_rel,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            cout<<"miss relation:"<<s3<<endl;
        }
		int e1 = entity2id[s1];
		int rel = relation2id[s3];
		int e2 = entity2id[s2];
		entity_rel_left[e1][rel] = 1;
		entity_rel_right[e2][rel] = 1;
		link_left[e1].push_back(make_pair(rel,e2));
		link_right[e2].push_back(make_pair(rel,e1));
    }
    fclose(f_rel);
	/*
	limit_left.resize(relation_num);
	FILE* f_left = fopen("limit/left.txt", "r");
	while ((fscanf(f_left,"%s",buf)==1))
	{
        string s1=buf;
        fscanf(f_left,"%s",buf);
        string s2=buf;
        fscanf(f_left,"%s",buf);
        string s3=buf;
        if (relation2id.count(s1)==0)
        {
            cout<<"miss relation:"<<s1<<endl;
        }
        if (attribute2id.count(s2)==0)
        {
            cout<<"miss attribute:"<<s2<<endl;
        }
        if (val2id.count(s3)==0)
        {
            cout<<"miss val:"<<s3<<endl;
        }
		int rel = relation2id[s1];
		int attr = attribute2id[s2];
		int val = val2id[s3];
		double f;
		fscanf(f_left,"%lf",&f);
		//if (f>0.99)
		limit_left[rel].push_back(attr_val(attr,val,f));
	}
	fclose(f_left);
	
	limit_right.resize(relation_num);
	FILE* f_right = fopen("limit/right.txt", "r");
	while ((fscanf(f_right,"%s",buf)==1))
	{
        string s1=buf;
        fscanf(f_right,"%s",buf);
        string s2=buf;
        fscanf(f_right,"%s",buf);
        string s3=buf;
        if (relation2id.count(s1)==0)
        {
            cout<<"miss relation:"<<s1<<endl;
        }
        if (attribute2id.count(s2)==0)
        {
            cout<<"miss attribute:"<<s2<<endl;
        }
        if (val2id.count(s3)==0)
        {
            cout<<"miss val:"<<s3<<endl;
        }
		int rel = relation2id[s1];
		int attr = attribute2id[s2];
		int val = val2id[s3];
		double f;
		fscanf(f_right,"%lf",&f);
		//if (f>0.99)
		limit_right[rel].push_back(attr_val(attr,val,f));
	}
	fclose(f_right);*/
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
	fclose(f_attr_correlation);
	/*FILE* f_left2right = fopen("limit/left2right.txt", "r");
	while ((fscanf(f_left2right,"%s",buf)==1))
	{
        string s1=buf;
        fscanf(f_left2right,"%s",buf);
        string s2=buf;
        fscanf(f_left2right,"%s",buf);
        string s3=buf;
        fscanf(f_left2right,"%s",buf);
        string s4=buf;
        fscanf(f_left2right,"%s",buf);
        string s5=buf;
		double pr;
		fscanf(f_attr_correlation, "%lf", &pr);
		int rel = relation2id[s1];
		int attr = attribute2id[s2], val = val2id[s3];
		int attr1 = attribute2id[s4], val1 = val2id[s5];
		left2right[rel][make_pair(attr1,val1)].push_back(attr_val(attr,val,pr));
	}
	fclose(f_attr_correlation);*/
}


int main(int argc,char**argv)
{
    if (argc<2)
        return 0;
    else
    {
        version = argv[1];
        prepare();
        test.run();
    }
}


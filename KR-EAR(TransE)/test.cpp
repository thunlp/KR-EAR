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
bool L1_flag=1;

#define EXP_TABLE_SIZE 1000000

double MAX_EXP = 6;
vector<double> expTable;


string version;
string trainortest = "test";

map<string,int> relation2id,entity2id, attribute2id, val2id;


vector<vector<pair<int,int> > > entity_attr_val;

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

vector<map<pair<int, int>, double> > limit_left, limit_right;

int relation_num,entity_num, attribute_num, val_num;
int n= 100;
int m = 100;

double sigmod(double x)
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

map<int, map<int,double> > cover;
int step, cover_num=0, cover_num1 = 0;

class Test{
    vector<vector<double> > relation_vec,entity_vec,attribute_vec, val_vec;

	vector<vector<vector<double> > > attribute_matrix;;


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
    double len;
	double sigmoid(double x)
	{
		if (x>MAX_EXP)
			return 1;
		else
			if (x<-MAX_EXP)
				return 0;
		else
			return expTable[(int)((x + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
			//return 1/(1+exp(-x));
	}
	double calc_attr(int e1, int val, int attr)
	{
	//	cout<<e1<<' '<<val<<' '<<rel<<endl;
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
			sum+=fabs(val_vec[val][ii]-syn1[ii]);
		return sigmoid((-sum + 7));
	}
	
    double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            sum+=-fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
        for (int ii=0; ii<n; ii++)
            sum+=-sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
		sum = sigmoid((sum + 7));
		/*double sum1 = 0;
		
		int id_num = entity_attr_val[e1].size();
		for (int k=0; k<id_num; k++)
		{
			int attr = entity_attr_val[e1][k].first;
			int val = entity_attr_val[e1][k].second;
			//double pr = attr_correlation[make_pair(attr1,val1)][make_pair(attr,val)];
			if (limit_left[rel].count(make_pair(attr,val)) > 0)
				sum1+= limit_left[rel][make_pair(attr,val)]/id_num  ;
			//sum1-=dis_attr(attr,val,attr1,val1)/id_num;
		}
		id_num = entity_attr_val[e2].size();
		for (int k=0; k<id_num; k++)
		{
			int attr = entity_attr_val[e2][k].first;
			int val = entity_attr_val[e2][k].second;
			//double pr = attr_correlation[make_pair(attr1,val1)][make_pair(attr,val)];
			if (limit_right[rel].count(make_pair(attr,val)) > 0)
				sum1+= limit_right[rel][make_pair(attr,val)]/id_num  ;
			//sum1-=dis_attr(attr,val,attr1,val1)/id_num;
		}
		sum+=0.3*sum1;*/

		//if (ok[make_pair(e1,rel)].count(e2)==1)
		//	cout<<sum1<<endl;
        return sum;
    }
    void run()
    {
		{
	        FILE* f1 = fopen(("relation2vec"+version+".txt").c_str(),"r");
	        FILE* f3 = fopen(("entity2vec"+version+".txt").c_str(),"r");
	        cout<<relation_num<<' '<<entity_num<<endl;
	        int relation_num_fb=relation_num;
	        relation_vec.resize(relation_num_fb);
	        for (int i=0; i<relation_num_fb;i++)
	        {
	            relation_vec[i].resize(n);
	            for (int ii=0; ii<n; ii++)
	                fscanf(f1,"%lf",&relation_vec[i][ii]);
	        }
	        entity_vec.resize(entity_num);
	        for (int i=0; i<entity_num;i++)
	        {
	            entity_vec[i].resize(n);
	            for (int ii=0; ii<n; ii++)
	                fscanf(f3,"%lf",&entity_vec[i][ii]);
	            if (vec_len(entity_vec[i])-1>1e-3)
	            	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
	        }
	        fclose(f1);
	        fclose(f3);
		}
		
        FILE* f1 = fopen(("attr2vec"+version+".txt").c_str(),"r");
        attribute_vec.resize(attribute_num);
        for (int i=0; i<attribute_num;i++)
        {
            attribute_vec[i].resize(m);
            for (int ii=0; ii<m; ii++)
                fscanf(f1,"%lf",&attribute_vec[i][ii]);
        }
        
        fclose(f1);
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
		
		cout<<"here"<<endl;
		double lsum=0 ,lsum_filter= 0;
		double rsum = 0,rsum_filter=0;
		double mid_sum = 0,mid_sum_filter=0;
		double lp_n=0,lp_n_filter = 0;
		double rp_n=0,rp_n_filter = 0;
		double mid_p_n=0,mid_p_n_filter = 0;
		map<int,double> lsum_r,lsum_filter_r;
		map<int,double> rsum_r,rsum_filter_r;
		map<int,double> mid_sum_r,mid_sum_filter_r;
		map<int,double> lp_n_r,lp_n_filter_r;
		map<int,double> rp_n_r,rp_n_filter_r;
		map<int,double> mid_p_n_r,mid_p_n_filter_r;
		map<int,int> rel_num;

		int hit_n = 10;
		int test_num =0;
		for (int testid = 0; testid<fb_l.size(); testid+=1)
		//for (int testid = 3; testid<4; testid+=1)
		{
			step = testid;
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
			double tmp = calc_sum(h,l,rel);
			rel_num[rel]+=1;
			vector<pair<int,double> > a;
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(i,l,rel);
				a.push_back(make_pair(i,sum));
			}
			if (cover.count(step)>0)
			{
				cover_num++;
				if (cover[step].count(h))
					cover_num1++;
				//else
				//	continue;
				//cout<<testid<<' '<<cover[step].size()<<endl;
			}
		//	else
		//		continue;
			sort(a.begin(),a.end(),cmp);
			double ttt=0;
			int filter = 0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(a[i].first,rel)].count(l)>0)
					ttt++;
			    if (ok[make_pair(a[i].first,rel)].count(l)==0)
			    	filter+=1;
				if (a[i].first ==h)
				{
					lsum+=a.size()-i;
					lsum_filter+=filter+1;
					lsum_r[rel]+=a.size()-i;
					lsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_n)
					{
						lp_n+=1;
						lp_n_r[rel]+=1;
					}	
					if (filter<hit_n)
					{
						lp_n_filter+=1;
						lp_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			a.clear();
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(h,i,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			ttt=0;
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{

				if (ok[make_pair(h,rel)].count(a[i].first)>0)
					ttt++;
				if (ok[make_pair(h,rel)].count(a[i].first)==0)
			    	filter+=1;
				if (a[i].first==l)
				{
					rsum+=a.size()-i;
					rsum_filter+=filter+1;
					rsum_r[rel]+=a.size()-i;
					rsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_n)
					{
						rp_n+=1;
						rp_n_r[rel]+=1;
					}
					if (filter<hit_n)
					{
						rp_n_filter+=1;
						rp_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			a.clear();
			for (int i=0; i<relation_num; i++)
			{
				double sum =calc_sum(h,l,i);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			ttt=0;
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(h,a[i].first)].count(l)>0)
					ttt++;
				if (ok[make_pair(h,a[i].first)].count(l)==0)
			    	filter+=1;
				if (a[i].first==rel)
				{
					mid_sum+=a.size()-i;
					mid_sum_filter+=filter+1;
					mid_sum_r[rel]+=a.size()-i;
					mid_sum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_n)
					{
						mid_p_n+=1;
						mid_p_n_r[rel]+=1;
					}
					if (filter<hit_n)
					{
						mid_p_n_filter+=1;
						mid_p_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			test_num+=1;
			if (test_num%100==0)
			{
				cout<<test_num<<":"<<endl;
				cout<<"left:"<<lsum/test_num<<'\t'<<lp_n/test_num<<"\t"<<lsum_filter/test_num<<'\t'<<lp_n_filter/test_num<<endl;
				cout<<"right:"<<rsum/test_num<<'\t'<<rp_n/test_num<<'\t'<<rsum_filter/test_num<<'\t'<<rp_n_filter/test_num<<endl;
				cout<<"mid:"<<mid_sum/test_num<<' '<<mid_p_n/test_num<<"\t"<<mid_sum_filter/test_num<<' '<<mid_p_n_filter/test_num<<endl;
				cout<<cover_num<<' '<<cover_num1<<endl;
			}
		}
		cout<<"left:"<<lsum/test_num<<'\t'<<lp_n/test_num<<"\t"<<lsum_filter/test_num<<'\t'<<lp_n_filter/test_num<<endl;
		cout<<"right:"<<rsum/test_num<<'\t'<<rp_n/test_num<<'\t'<<rsum_filter/test_num<<'\t'<<rp_n_filter/test_num<<endl;
		cout<<"mid:"<<mid_sum/test_num<<' '<<mid_p_n/test_num<<"\t"<<mid_sum_filter/test_num<<' '<<mid_p_n_filter/test_num<<endl;
    }

};
Test test;

void prepare()
{
    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/relation2id.txt","r");
	FILE* f3 = fopen("../data/attribute2id.txt","r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		relation_num++;
	}
	while (fscanf(f3,"%s%d",buf,&x)==2)
	{
		string st=buf;
		attribute2id[st]=x;
		attribute_num++;
	}
	fclose(f1);
	fclose(f2);
	fclose(f3);
	FILE* f4 = fopen("../data/val2id.txt", "r");
	while (fscanf(f4,"%s%d",buf,&x)==2)
	{
		string st=buf;
		val2id[st]=x;
		val_num++;
	}
	cout<<entity_num<<' '<<relation_num<<' '<<attribute_num<<' '<<val_num<<endl;
    FILE* f_kb = fopen("../data/test-rel.txt","r");
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
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
        	cout<<"miss relation:"<<s3<<endl;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
    }
    fclose(f_kb);
    FILE* f_kb1 = fopen("../data/train-rel.txt","r");
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
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            cout<<"miss relation:"<<s3<<endl;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
	entity_attr_val.resize(entity_num);
    FILE* f_attr = fopen("../data/train-attr.txt","r");
	while (fscanf(f_attr,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_attr,"%s",buf);
        string s2=buf;
        fscanf(f_attr,"%s",buf);
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

		entity_attr_val[e1].push_back(make_pair(attr,val));
    }
    fclose(f_attr);
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
		limit_left[rel][make_pair(attr,val)] = f;
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
		limit_right[rel][make_pair(attr,val)]=f;
	}
	
}


int main(int argc,char**argv)
{
    if (argc<2)
        return 0;
    else
    {
        version = argv[1];
		for (int i = 0; i < EXP_TABLE_SIZE; i++) 
		{
			double tmp = exp((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
			expTable.push_back(tmp/(tmp+1));                   // Precompute f(x) = x / (x + 1)
		}
        prepare();
        test.run();
    }
}


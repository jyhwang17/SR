import numpy
import pandas
import pickle
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from itertools import chain
from scipy.sparse import csr_matrix
import collections
from collections import defaultdict


def load_cache(load_dir_path, name):
    
    with open(load_dir_path+name+".pkl",'rb') as f:
        data_dict = pickle.load(f)
    
    return data_dict

class InteractionAnalyzer():

    def __init__(self, load_dir_path, name):
        
        #dataframe
        try:
            self.df = load_cache(load_dir_path, 'filtered_df')
        except Exception as e:
            print('load error',e)
    
        self.user_key = 'user_id'
        self.item_key = 'problem_id'
        self.time_key = 'in_date'
        self.num_users = self.get_num_users() 
        self.num_items = self.get_num_items()
        self.max_user_id = self.df[self.user_key].max()
        self.max_item_id = self.df[self.item_key].max()
        
    def update_df(self, df):
        
        self.df = df
        
    def set_user_key(self, user_key = 'user_id'):
        
        self.user_key = user_key
    
    def set_item_key(self, item_key = 'problem_id'):
        
        self.item_key = item_key
    
    def get_num_users(self):
        
        return len(self.df[self.user_key].unique())
        
    def get_num_items(self):
        
        return len(self.df[self.item_key].unique())
        
    def get_user_unique_engagement(self) -> pandas.core.series.Series:
        
        return self.df.groupby(self.user_key)[self.item_key].unique().agg(len)
    
    def get_user_engagement(self) -> pandas.core.series.Series:
        
        return self.df.groupby(self.user_key)[self.item_key].agg(len)
    
    def get_user_stats(self) -> collections.defaultdict:
        
        # Compute user_stats
        # The user_stats contains (the engagement, z-score, ranking(percent), item(percent) )
        
        user_engagement = self.get_user_unique_engagement()
        sorted_engagement= user_engagement.sort_values().values
        
        eng_standard = (user_engagement.values - user_engagement.mean() ) / user_engagement.std() #정규분포
        eng_rank = self.num_users - np.searchsorted(sorted_engagement, user_engagement.values) -1
        
        user_stats = defaultdict(tuple)
        
        for (user_id_, eng_, eng_stand_, eng_rank_) in zip(user_engagement.keys(), 
                                                          user_engagement.values, 
                                                          eng_standard, 
                                                          eng_rank):
            
            user_stats[user_id_] = (eng_, eng_stand_, (eng_rank_/self.num_users), eng_/self.num_items)
        
        return user_stats
    
    def get_item_unique_popularity(self) -> pandas.core.series.Series:
        
        return self.df.groupby(self.item_key)[self.user_key].unique().agg(len)
    
    def get_item_popularity(self) -> pandas.core.series.Series:
        
        return self.df.groupby(self.item_key)[self.user_key].agg(len)
    
    def get_item_stats(self)  -> collections.defaultdict:
        
        # Compute item_stats
        # The item_stats contains (the popularity, z-score, ranking(percent), user(percent) )
        
        item_popularity = self.get_item_unique_popularity()
        sorted_popularity = item_popularity.sort_values().values
        
        pop_standard = (item_popularity.values - item_popularity.mean() ) / item_popularity.std() #정규분포
        pop_rank = self.num_items - np.searchsorted(sorted_popularity, item_popularity.values) -1 
        
        item_stats = defaultdict(tuple)
        
        for (item_id_, pop_, pop_stand_, pop_rank_) in zip(item_popularity.keys(),
                                                           item_popularity.values,
                                                           pop_standard,
                                                           pop_rank):
            
            item_stats[item_id_] = (pop_, pop_stand_, (pop_rank_/self.num_items), pop_/self.num_users)
  
        return item_stats
    
    def get_matrix_sparsity(self) -> numpy.float64:

        return 1.0 - self.df.groupby(self.user_key)[self.item_key].unique().agg(len).agg(sum)/(self.num_users*self.num_items)
    
    def get_transition_coo(self) -> numpy.ndarray:
        
        seq = self.df.sort_values(by=self.time_key).groupby(self.user_key)[self.item_key].apply(np.array)# Build sequential Interaction
        
        def drop_consecutive_duplicates(seq):
            return seq[np.concatenate(([True], seq[1:] != seq[:-1] ))]

        seq = seq.apply(lambda x: drop_consecutive_duplicates(x))# Drop consecutive_duplicates
        
        seq_coo = seq.apply(lambda x: list( zip(x,x[1:]) ) )# interaction의 sequence를 transition pair로 변환

        coo = list(chain.from_iterable(seq_coo))
        coo = np.array(list(map(lambda coo: list(coo), coo)))
        
        return coo
    
    def get_transition_stats(self) -> collections.defaultdict:
        
        coo = self.get_transition_coo()
        transition_popularity = defaultdict(int)
        
        for [row,col] in coo:
            transition_popularity[(row,col)] += 1
            
        sorted_transition_array = np.array(list(sorted(transition_popularity.values(), key = lambda x: x )))
        transition_array = np.array(list(transition_popularity.values()))
                
        pop_standard = (transition_array - transition_array.mean()) / transition_array.std()

        pop_rank = len(transition_array) - np.searchsorted(sorted_transition_array, transition_array) -1
                
        transition_stats = defaultdict(tuple)

        
        for ( (from_,to_), pop_, pop_stand_, pop_rank_) in zip(transition_popularity.keys(),
                                                               transition_array,
                                                               pop_standard,
                                                               pop_rank):
            
            transition_stats[(from_,to_)] = (pop_, pop_stand_, (pop_rank_/len(transition_array)) )
        
        return transition_stats
    
    def get_sequence_item_popularity(self) -> collections.defaultdict:
        
        coo = self.get_transition_coo()
        sequence_item_popularity = defaultdict(int)
        for [row,col] in coo:
            sequence_item_popularity[row]+=1.
            sequence_item_popularity[col]+=1.
            
        return sequence_item_popularity
    
    def get_transition_matrix(self) -> scipy.sparse.csr.csr_matrix:
        
        # We assume max_item_id is integer type
        coo = self.get_transition_coo()
        
        return csr_matrix(( np.ones_like(coo[:,0]), (coo[:,0], coo[:,1])), shape=(self.max_item_id+1, self.max_item_id+1))

    def show_cum_popularity_distribution(self, title='Cumulative Item Popularity'):
        
        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=14)     # fontsize of the axes title
        plt.rcParams['axes.grid']= True
    
        fig,ax = plt.subplots()
        sorted_popularity = np.array(sorted(list(self.get_item_unique_popularity()))) 
        ax.bar(np.arange(len(sorted_popularity)),sorted_popularity.cumsum(0), width=1.)
        ax.ticklabel_format(axis='y',style='plain')
        ax.set_title(title)
    
        # Turn off tick labels
        ax.set_ylabel('Cumulative interactions')
        plt.show()
    
    def show_cum_engagement_distribution(self, title='Cumulative User Engagement'):
        
        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=14)     # fontsize of the axes title
        plt.rcParams['axes.grid']= True
    
        fig,ax = plt.subplots()
        sorted_engagement = np.array(sorted(list(self.get_user_unique_engagement()) ) ) 
        ax.bar(np.arange(len(sorted_engagement)),sorted_engagement.cumsum(0), width=1.)
        ax.ticklabel_format(axis='y',style='plain')
        ax.set_title(title)
    
        # Turn off tick labels
        ax.set_ylabel('Cumulative interactions')
        plt.show()
    
    def show_cum_transition_distribution(self, title='Cumultative Transition', option='normalize'):
        
        coo = self.get_transition_coo()
        transition_popularity = defaultdict(int)
        sequence_item_popularity = self.get_sequence_item_popularity()
        if option =='normalize':
            for [row,col] in coo:
                transition_popularity[(row,col)] += 1/(np.sqrt(sequence_item_popularity[row])*np.sqrt(sequence_item_popularity[col]))
        else:
            for [row,col] in coo:
                transition_popularity[(row,col)] += 1.
                
        transition_popularity = list(transition_popularity.values())
        sorted_transition = np.array(sorted(transition_popularity))
        
        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=14)     # fontsize of the axes title
        plt.rcParams['axes.grid']= True
    
        fig,ax = plt.subplots()
        ax.bar(np.arange(len(sorted_transition)),sorted_transition.cumsum(0), width=1.)
        ax.ticklabel_format(axis='y',style='plain')
        ax.set_title(title)
    
        # Turn off tick labels
        if option == 'normalize':
            ax.set_ylabel('Cumulative normalized interactions')
        else:
            ax.set_ylabel('Cumulative interactions')
        plt.show()
    
    def show_cumulative_distribution(self, popularity, title = ''):
    
        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=14)     # fontsize of the axes title
        plt.rcParams['axes.grid']= True
    
        fig,ax = plt.subplots()
        sorted_popularity = np.array(sorted(popularity)) 
        ax.bar(np.arange(len(sorted_popularity)),sorted_popularity.cumsum(0), width=1.)
        ax.ticklabel_format(axis='y',style='plain')
        ax.set_title(title)
    
        # Turn off tick labels
        ax.set_ylabel('Cumulative interactions')
        plt.show()
        
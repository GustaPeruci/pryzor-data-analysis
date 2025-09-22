import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configurar o estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SteamGameAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.app_info = None
        self.price_data = {}
        self.processed_data = None
        self.features_data = None
        
    def load_application_data(self):
        """Etapa 1: Carregamento e exploração inicial dos dados"""
        print("=== ETAPA 1: EXPLORANDO ESTATÍSTICAS DESCRITIVAS ===")
        
        # Carregar dados das aplicações
        self.app_info = pd.read_csv(os.path.join(self.data_path, 'applicationInformation.csv'), encoding='latin-1')
        
        print(f"Dimensões do dataset principal: {self.app_info.shape}")
        print(f"\nColunas: {list(self.app_info.columns)}")
        print(f"\nTipos de dados:")
        print(self.app_info.dtypes)
        
        print(f"\nPrimeiras 5 linhas:")
        print(self.app_info.head())
        
        print(f"\nInformações estatísticas básicas:")
        print(self.app_info.describe())
        
        # Verificar valores únicos
        print(f"\nTipos de aplicações: {self.app_info['type'].value_counts()}")
        print(f"\nJogos gratuitos vs pagos: {self.app_info['freetoplay'].value_counts()}")
        
        return self.app_info
    
    def load_price_history_sample(self, sample_size=50):
        """Carregar uma amostra dos dados de histórico de preços"""
        print(f"\n=== CARREGANDO AMOSTRA DE {sample_size} ARQUIVOS DE PREÇOS ===")
        
        price_files = glob.glob(os.path.join(self.data_path, 'PriceHistory', '*.csv'))
        
        # Pegar uma amostra dos arquivos
        sample_files = price_files[:sample_size]
        
        for file_path in sample_files:
            app_id = os.path.basename(file_path).replace('.csv', '')
            try:
                price_df = pd.read_csv(file_path, encoding='latin-1')
                self.price_data[app_id] = price_df
            except Exception as e:
                print(f"Erro ao carregar {file_path}: {e}")
                
        print(f"Carregados {len(self.price_data)} arquivos de preços")
        
        # Analisar estrutura de um arquivo exemplo
        if self.price_data:
            sample_key = list(self.price_data.keys())[0]
            sample_df = self.price_data[sample_key]
            print(f"\nEstrutura de dados de preço (App ID {sample_key}):")
            print(f"Dimensões: {sample_df.shape}")
            print(f"Colunas: {list(sample_df.columns)}")
            print(f"Período: {sample_df['Date'].min()} a {sample_df['Date'].max()}")
            print(sample_df.head())
            
        return self.price_data
    
    def data_cleaning(self):
        """Etapa 2: Limpeza de dados"""
        print("\n=== ETAPA 2: LIMPEZA DE DADOS ===")
        
        # Limpar dados das aplicações
        print("Limpando dados das aplicações...")
        initial_shape = self.app_info.shape
        
        # Verificar valores ausentes
        print(f"Valores ausentes por coluna:")
        print(self.app_info.isnull().sum())
        
        # Remover linhas com informações essenciais ausentes
        self.app_info = self.app_info.dropna(subset=['appid', 'name'])
        
        # Preencher valores ausentes em 'type' com 'unknown'
        self.app_info['type'] = self.app_info['type'].fillna('unknown')
        
        # Preencher valores ausentes em 'releasedate' 
        self.app_info['releasedate'] = self.app_info['releasedate'].fillna('Unknown')
        
        # Preencher valores ausentes em 'freetoplay' com 0 (assumindo que é pago)
        self.app_info['freetoplay'] = self.app_info['freetoplay'].fillna(0)
        
        # Remover duplicatas baseadas no appid
        self.app_info = self.app_info.drop_duplicates(subset=['appid'])
        
        print(f"Dados limpos - shape inicial: {initial_shape}, shape final: {self.app_info.shape}")
        
        # Limpar dados de preços
        print("\nLimpando dados de preços...")
        cleaned_price_data = {}
        
        for app_id, price_df in self.price_data.items():
            # Remover linhas com valores ausentes
            cleaned_df = price_df.dropna()
            
            # Verificar se há dados suficientes após limpeza
            if len(cleaned_df) > 10:  # Manter apenas se tiver pelo menos 10 registros
                cleaned_price_data[app_id] = cleaned_df
                
        self.price_data = cleaned_price_data
        print(f"Mantidos {len(self.price_data)} arquivos de preços após limpeza")
        
        return self.app_info, self.price_data
    
    def data_transformation(self):
        """Etapa 3: Transformação de variáveis"""
        print("\n=== ETAPA 3: TRANSFORMAÇÃO DE VARIÁVEIS ===")
        
        # Transformar dados das aplicações
        print("Transformando dados das aplicações...")
        
        # Converter releasedate para datetime
        def parse_date(date_str):
            if pd.isna(date_str) or date_str == 'Unknown':
                return None
            try:
                return pd.to_datetime(date_str, format='%d-%b-%y')
            except:
                return None
                
        self.app_info['release_date_parsed'] = self.app_info['releasedate'].apply(parse_date)
        
        # Extrair ano de lançamento
        self.app_info['release_year'] = self.app_info['release_date_parsed'].dt.year
        
        # Codificar variáveis categóricas
        le_type = LabelEncoder()
        self.app_info['type_encoded'] = le_type.fit_transform(self.app_info['type'].astype(str))
        
        # Transformar dados de preços
        print("Transformando dados de preços...")
        transformed_price_data = {}
        
        for app_id, price_df in self.price_data.items():
            df_copy = price_df.copy()
            
            # Converter Date para datetime
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            
            # Extrair características temporais
            df_copy['year'] = df_copy['Date'].dt.year
            df_copy['month'] = df_copy['Date'].dt.month
            df_copy['day_of_week'] = df_copy['Date'].dt.dayofweek
            df_copy['quarter'] = df_copy['Date'].dt.quarter
            
            # Calcular economia absoluta
            df_copy['savings_amount'] = df_copy['Initialprice'] - df_copy['Finalprice']
            
            # Indicador de desconto binário
            df_copy['has_discount'] = (df_copy['Discount'] > 0).astype(int)
            
            transformed_price_data[app_id] = df_copy
            
        self.price_data = transformed_price_data
        print("Transformações aplicadas com sucesso!")
        
        return self.app_info, self.price_data
    
    def feature_engineering(self):
        """Etapa 4: Engenharia de recursos"""
        print("\n=== ETAPA 4: ENGENHARIA DE RECURSOS ===")
        
        features_list = []
        
        for app_id, price_df in self.price_data.items():
            # Encontrar informações do jogo
            app_info_row = self.app_info[self.app_info['appid'] == int(app_id)]
            
            if len(app_info_row) == 0:
                continue
                
            app_info_row = app_info_row.iloc[0]
            
            # Calcular estatísticas de preço
            price_stats = {
                'app_id': int(app_id),
                'game_name': app_info_row['name'],
                'game_type': app_info_row['type'],
                'is_free_to_play': app_info_row['freetoplay'],
                'release_year': app_info_row['release_year'],
                
                # Estatísticas de preço
                'avg_initial_price': price_df['Initialprice'].mean(),
                'min_initial_price': price_df['Initialprice'].min(),
                'max_initial_price': price_df['Initialprice'].max(),
                'price_volatility': price_df['Initialprice'].std(),
                
                'avg_final_price': price_df['Finalprice'].mean(),
                'min_final_price': price_df['Finalprice'].min(),
                'max_final_price': price_df['Finalprice'].max(),
                
                # Estatísticas de desconto
                'avg_discount': price_df['Discount'].mean(),
                'max_discount': price_df['Discount'].max(),
                'discount_frequency': (price_df['Discount'] > 0).mean(),
                'total_discount_days': (price_df['Discount'] > 0).sum(),
                
                # Características temporais
                'total_observation_days': len(price_df),
                'price_trend': self._calculate_price_trend(price_df),
                'seasonal_discount_pattern': self._analyze_seasonal_discounts(price_df),
                
                # Indicadores de melhor momento para compra
                'best_discount_month': price_df.groupby('month')['Discount'].mean().idxmax(),
                'worst_discount_month': price_df.groupby('month')['Discount'].mean().idxmin(),
                
                # Economia potencial
                'max_savings': price_df['savings_amount'].max(),
                'avg_savings_when_discounted': price_df[price_df['Discount'] > 0]['savings_amount'].mean() if (price_df['Discount'] > 0).any() else 0
            }
            
            features_list.append(price_stats)
        
        self.features_data = pd.DataFrame(features_list)
        
        # Tratar valores ausentes nas features criadas
        self.features_data = self.features_data.fillna(0)
        
        print(f"Features criadas para {len(self.features_data)} jogos")
        print(f"Total de features: {len(self.features_data.columns)}")
        print("\nPrimeiras 5 linhas das features:")
        print(self.features_data.head())
        
        return self.features_data
    
    def _calculate_price_trend(self, price_df):
        """Calcular tendência de preço ao longo do tempo"""
        if len(price_df) < 2:
            return 0
        
        # Correlação simples entre tempo e preço
        price_df_sorted = price_df.sort_values('Date')
        time_numeric = range(len(price_df_sorted))
        correlation = np.corrcoef(time_numeric, price_df_sorted['Initialprice'])[0, 1]
        
        return correlation if not np.isnan(correlation) else 0
    
    def _analyze_seasonal_discounts(self, price_df):
        """Analisar padrões sazonais de desconto"""
        monthly_discounts = price_df.groupby('month')['Discount'].mean()
        seasonal_variance = monthly_discounts.var()
        
        return seasonal_variance if not np.isnan(seasonal_variance) else 0
    
    def data_splitting(self, test_size=0.2, val_size=0.1):
        """Etapa 5: Divisão de dados"""
        print("\n=== ETAPA 5: DIVISÃO DE DADOS ===")
        
        if self.features_data is None:
            print("Erro: Execute feature_engineering() primeiro")
            return None
        
        # Separar features e variáveis categóricas
        numeric_features = self.features_data.select_dtypes(include=[np.number]).columns
        categorical_features = self.features_data.select_dtypes(exclude=[np.number]).columns
        
        print(f"Features numéricas: {len(numeric_features)}")
        print(f"Features categóricas: {len(categorical_features)}")
        
        # Para este projeto, vamos criar um target sintético baseado em características
        # de "melhor momento para compra"
        self.features_data['good_buy_time'] = self._create_target_variable()
        
        # Preparar dados para divisão
        X = self.features_data[numeric_features].drop(['good_buy_time'], axis=1, errors='ignore')
        y = self.features_data['good_buy_time']
        
        # Primeira divisão: treino+validação vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Segunda divisão: treino vs validação
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Divisão dos dados:")
        print(f"  Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X):.1%})")
        print(f"  Validação: {X_val.shape[0]} amostras ({X_val.shape[0]/len(X):.1%})")
        print(f"  Teste: {X_test.shape[0]} amostras ({X_test.shape[0]/len(X):.1%})")
        
        # Normalizar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Converter de volta para DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        datasets = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': list(X.columns)
        }
        
        return datasets
    
    def _create_target_variable(self):
        """Criar variável target para 'melhor momento para compra'"""
        # Critérios para definir um "bom momento para compra":
        # 1. Alta frequência de desconto
        # 2. Descontos significativos
        # 3. Preço abaixo da média
        
        conditions = []
        
        # Condição 1: Frequência de desconto acima da mediana
        discount_freq_median = self.features_data['discount_frequency'].median()
        conditions.append(self.features_data['discount_frequency'] > discount_freq_median)
        
        # Condição 2: Desconto médio acima da mediana
        avg_discount_median = self.features_data['avg_discount'].median()
        conditions.append(self.features_data['avg_discount'] > avg_discount_median)
        
        # Condição 3: Economia máxima acima da mediana
        max_savings_median = self.features_data['max_savings'].median()
        conditions.append(self.features_data['max_savings'] > max_savings_median)
        
        # Um jogo é considerado "bom para comprar" se atender pelo menos 2 condições
        good_buy = sum(conditions) >= 2
        
        return good_buy.astype(int)
    
    def generate_visualizations(self):
        """Gerar visualizações dos dados processados"""
        print("\n=== GERANDO VISUALIZAÇÕES ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análise Exploratória - Dados de Jogos Steam', fontsize=16)
        
        # 1. Distribuição de preços inicial
        axes[0, 0].hist(self.features_data['avg_initial_price'], bins=50, alpha=0.7)
        axes[0, 0].set_title('Distribuição de Preços Iniciais')
        axes[0, 0].set_xlabel('Preço Médio Inicial')
        axes[0, 0].set_ylabel('Frequência')
        
        # 2. Frequência de desconto vs Preço
        axes[0, 1].scatter(self.features_data['avg_initial_price'], 
                          self.features_data['discount_frequency'], alpha=0.6)
        axes[0, 1].set_title('Preço vs Frequência de Desconto')
        axes[0, 1].set_xlabel('Preço Médio Inicial')
        axes[0, 1].set_ylabel('Frequência de Desconto')
        
        # 3. Distribuição de descontos máximos
        axes[0, 2].hist(self.features_data['max_discount'], bins=30, alpha=0.7)
        axes[0, 2].set_title('Distribuição de Descontos Máximos')
        axes[0, 2].set_xlabel('Desconto Máximo (%)')
        axes[0, 2].set_ylabel('Frequência')
        
        # 4. Análise temporal - Ano de lançamento
        release_year_counts = self.features_data['release_year'].value_counts().sort_index()
        axes[1, 0].plot(release_year_counts.index, release_year_counts.values)
        axes[1, 0].set_title('Jogos por Ano de Lançamento')
        axes[1, 0].set_xlabel('Ano de Lançamento')
        axes[1, 0].set_ylabel('Número de Jogos')
        
        # 5. Correlação entre características
        corr_features = ['avg_initial_price', 'discount_frequency', 'max_discount', 'price_volatility']
        corr_matrix = self.features_data[corr_features].corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Matriz de Correlação')
        axes[1, 1].set_xticks(range(len(corr_features)))
        axes[1, 1].set_yticks(range(len(corr_features)))
        axes[1, 1].set_xticklabels(corr_features, rotation=45)
        axes[1, 1].set_yticklabels(corr_features)
        plt.colorbar(im, ax=axes[1, 1])
        
        # 6. Target variable distribution
        target_counts = self.features_data['good_buy_time'].value_counts()
        axes[1, 2].bar(['Momento Ruim', 'Bom Momento'], target_counts.values)
        axes[1, 2].set_title('Distribuição da Variável Target')
        axes[1, 2].set_ylabel('Número de Jogos')
        
        plt.tight_layout()
        plt.savefig('steam_games_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_processed_data(self):
        """Salvar dados processados"""
        print("\n=== SALVANDO DADOS PROCESSADOS ===")
        
        # Salvar dados das aplicações limpos
        self.app_info.to_csv('processed_application_info.csv', index=False)
        
        # Salvar features engineered
        self.features_data.to_csv('engineered_features.csv', index=False)
        
        # Salvar resumo do processamento
        summary = {
            'total_games_original': len(self.app_info),
            'total_games_with_price_data': len(self.features_data),
            'total_features_created': len(self.features_data.columns),
            'data_processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('processing_summary.txt', 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        print("Dados processados salvos com sucesso!")
        print(f"Arquivos criados:")
        print(f"  - processed_application_info.csv")
        print(f"  - engineered_features.csv")
        print(f"  - processing_summary.txt")
        print(f"  - steam_games_analysis.png")
    
    def run_complete_analysis(self):
        """Executar análise completa"""
        print("INICIANDO ANÁLISE COMPLETA DE DADOS STEAM")
        print("=" * 50)
        
        # Etapa 1: Exploração
        self.load_application_data()
        self.load_price_history_sample(sample_size=100)  # Aumentar amostra
        
        # Etapa 2: Limpeza
        self.data_cleaning()
        
        # Etapa 3: Transformação
        self.data_transformation()
        
        # Etapa 4: Engenharia de Features
        self.feature_engineering()
        
        # Etapa 5: Divisão dos dados
        datasets = self.data_splitting()
        
        # Visualizações
        self.generate_visualizations()
        
        # Salvar resultados
        self.save_processed_data()
        
        print("\n" + "=" * 50)
        print("ANÁLISE COMPLETA FINALIZADA COM SUCESSO!")
        print("=" * 50)
        
        return datasets

# Executar análise
if __name__ == "__main__":
    # Definir caminho dos dados
    data_path = r"C:\Users\Gustavo\Documents\Projetos\Pryzor\Análise de dados"
    
    # Criar instância da análise
    analyzer = SteamGameAnalysis(data_path)
    
    # Executar análise completa
    datasets = analyzer.run_complete_analysis()
    
    # Exibir informações finais
    if datasets:
        print(f"\nDatasets criados:")
        print(f"  Treino: {datasets['X_train'].shape}")
        print(f"  Validação: {datasets['X_val'].shape}")
        print(f"  Teste: {datasets['X_test'].shape}")
        print(f"\nFeatures disponíveis: {len(datasets['feature_names'])}")
        print("Dados prontos para modelagem!")
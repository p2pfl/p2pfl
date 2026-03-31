import argparse
import os



os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import xgboost
from datasets import DatasetDict, Dataset
from p2pfl.learning.aggregators.fedxgbcyclic import FedXgbCyclic
from p2pfl.learning.frameworks.xgboost.xgboost_learner import XGBoostLearner
from p2pfl.learning.frameworks.xgboost.xgboost_model import XGBoostModel
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy, DirichletPartitionStrategy
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings, wait_convergence, wait_to_finish
from p2pfl.examples.cli.models.tabular.tabular_pytorch import model_build_fn as pytorch_model_build_fn



def set_data_train_test_split(args):
    if args.dataset:
        dataset_name = args.dataset.lower()
        base_dir = os.path.join(os.path.dirname(__file__), 'datasets')
        if dataset_name == 'ton-iot':
            dataset_path = os.path.join(base_dir, 'ton-iot', 'train_test_network.csv')
            df = pd.read_csv(dataset_path)

            # Drop columns that contain '-' in any of their values
            cols_with_dash = [col for col in df.columns if df[col].astype(str).str.contains('-').any()]
            # Drop src_ip and dst_ip
            cols_to_drop = cols_with_dash + ['src_ip', 'dst_ip']
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

            # Label encoding on 'proto' and 'conn_state'
            for col in ['proto', 'conn_state']:
                if col in df.columns:
                    le_col = LabelEncoder()
                    df[col] = le_col.fit_transform(df[col].astype(str))

            # Label encoding on 'type' column
            le = LabelEncoder()
            df['label'] = le.fit_transform(df['type'])
            label_keys = list(le.classes_)

            feature_keys = [col for col in df.columns if col not in ['type', 'label']]

            # Drop 'type' but keep 'label'
            df = df.drop(columns=['type'])

            dataset = Dataset.from_pandas(df)
            split = dataset.train_test_split(test_size=0.2, seed=args.seed)
            dataset = DatasetDict({"train": split["train"], "test": split["test"]})

            return dataset, label_keys, feature_keys
        elif dataset_name == 'ugr-16':
            train_path = os.path.join(base_dir, 'ugr-16', 'train.csv')
            test_path = os.path.join(base_dir, 'ugr-16', 'test.csv')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            dataset = DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df)
            })
            df = train_df
            target_col = args.target_name
            label_names = df[target_col].unique() if target_col in df.columns else None
            feature_keys = [col for col in df.columns if col != target_col]
            return dataset, label_names, feature_keys
        elif dataset_name == 'unr-idd':
            dataset_path = os.path.join(base_dir, 'unr-idd', 'UNR-IDD.csv')
            df = pd.read_csv(dataset_path)
            # Preprocesamiento específico para unr-idd
            df.drop(columns=['Switch ID'], inplace=True)
            df['Port Number'] = LabelEncoder().fit_transform(df['Port Number'].astype(str))
            df['is_valid'] = LabelEncoder().fit_transform(df['is_valid'].astype(str))
            df['Label'] = LabelEncoder().fit_transform(df['Label'].astype(str))
            df.drop(columns=['Binary Label'], inplace=True)
            label_names = df['Label'].unique().tolist()
            feature_keys = [col for col in df.columns if col not in ['Label', 'label']]
            dataset = Dataset.from_pandas(df)
            split = dataset.train_test_split(test_size=0.2, seed=args.seed)
            dataset = DatasetDict({"train": split["train"], "test": split["test"]})
            return dataset, label_names, feature_keys
        elif dataset_name == 'unsw-nb15':
            train_path = os.path.join(base_dir, 'unsw-nb15', 'UNSW_NB15_training-set.csv')
            test_path = os.path.join(base_dir, 'unsw-nb15', 'UNSW_NB15_testing-set.csv')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # Preprocesamiento específico para unsw-nb15
            for df in [train_df, test_df]:
                df.drop(columns=['id'], inplace=True)
                df['proto'] = LabelEncoder().fit_transform(df['proto'].astype(str))
                df.drop(columns=['service'], inplace=True)
                df['state'] = LabelEncoder().fit_transform(df['state'].astype(str))
                df['attack_cat'] = LabelEncoder().fit_transform(df['attack_cat'].astype(str))
                df.drop(columns=['label'], inplace=True)
            dataset = DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df)
            })
            label_names = train_df['attack_cat'].unique().tolist()
            feature_keys = [col for col in train_df.columns if col != 'attack_cat']
            return dataset, label_names, feature_keys
        elif dataset_name == 'zeekdata':
            zeek_dir = os.path.join(base_dir, 'zeekdata')
            csvs = [os.path.join(zeek_dir, f) for f in os.listdir(zeek_dir) if f.endswith('.csv')]
            if not csvs:
                raise ValueError("No se encontraron archivos CSV en zeekdata")
            csvs.sort()  # Ordenar alfabéticamente
            df = pd.read_csv(csvs[0])  # Solo usar el primer archivo CSV TODO: esto es porque tengo poca memoria
            df.dropna(inplace=True)
            target_categories = df['mitre_attack_tactics'].unique().tolist()
            # Drop unnecessary columns
            drop_cols = ['uid', 'datetime', 'src_ip', 'dest_ip', 'community_id', 'ts', 'duration']
            df = df.drop(columns=[col for col in drop_cols if col in df.columns])
            # Encode boolean columns
            for bool_col in ['local_resp', 'local_orig']:
                if bool_col in df.columns:
                    df[bool_col] = df[bool_col].astype(bool).astype(int)
            # Encode categorical columns
            categorical_cols = ['service', 'protocol', 'conn_state', 'history']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            # Encode the target label
            df['label'] = LabelEncoder().fit_transform(df['mitre_attack_tactics'])
            label_keys = target_categories
            feature_keys = [col for col in df.columns if col not in ['mitre_attack_tactics', 'label']]
            df = df.drop(columns=['mitre_attack_tactics'])
            dataset = Dataset.from_pandas(df)
            split = dataset.train_test_split(test_size=0.2, seed=args.seed)
            dataset = DatasetDict({"train": split["train"], "test": split["test"]})
            return dataset, label_keys, feature_keys
        else:
            raise ValueError(f"Dataset '{args.dataset}' no soportado.")
        dataset = Dataset.from_pandas(df)
        split = dataset.train_test_split(test_size=0.2, seed=args.seed)
        dataset = DatasetDict({"train": split["train"], "test": split["test"]})
        target_col = args.target_name
        label_names = df[target_col].unique() if target_col in df.columns else None
        feature_keys = [col for col in df.columns if col != target_col]
        return dataset, label_names, feature_keys
    if args.data_csv:
        # Cargar un solo archivo y generar split
        df = pd.read_csv(args.data_csv)
        dataset = Dataset.from_pandas(df)
        # Usar HuggingFace para split
        split = dataset.train_test_split(test_size=0.2, seed=args.seed)
        dataset = DatasetDict({"train": split["train"], "test": split["test"]})
    elif args.train_csv and args.test_csv:
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df)
        })
        df = train_df
    else:
        raise ValueError("Debes especificar --data_csv o ambos --train_csv y --test_csv")
    target_col = args.target_name
    label_names = df[target_col].unique() if target_col in df.columns else None
    feature_keys = [col for col in df.columns if col != target_col]
    return dataset, label_names, feature_keys


def run_federated(args):
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker()
    tracker.start()
    n = args.nodes
    r = args.rounds
    seed = args.seed
    topology = getattr(TopologyType, args.topology.upper())
    protocol_type = args.protocol.lower()
    framework = args.framework.lower()

    set_seed(seed)
    set_standalone_settings()

    data, label_names, feature_keys = set_data_train_test_split(args)
    data = P2PFLDataset(data)
    data.set_batch_size(64)  # batch size fijo
    if args.partition_strategy.lower() == 'randomiid':
        partition_strategy = RandomIIDPartitionStrategy
    elif args.partition_strategy.lower() == 'dirichlet':
        partition_strategy = DirichletPartitionStrategy
    else:
        raise ValueError("Estrategia de particionado no soportada. Usa 'randomiid' o 'dirichlet'.")
    partitions = data.generate_partitions(n, partition_strategy, label_tag=args.target_name)

    nodes = []
    for i in range(n):
        address = f"node-{i}" if protocol_type == "memory" else f"unix:///tmp/p2pfl-{i}.sock" if protocol_type == "unix" else "127.0.0.1"
        # Crear una instancia nueva del protocolo para cada nodo
        if protocol_type == "memory":
            from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
            comm_protocol = MemoryCommunicationProtocol()
        elif protocol_type == "grpc":
            from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
            comm_protocol = GrpcCommunicationProtocol()
        else:
            raise ValueError("Protocolo de comunicación no soportado. Usa 'memory' o 'grpc'.")

        # Instanciar agregador por nodo
        if framework == "xgboost":
            if args.agregator == "fedxgbcyclic":
                aggregator = FedXgbCyclic()
            elif args.agregator == "fedxgbbagging":
                from p2pfl.learning.aggregators.fedxgbbagging import FedXgbBagging
                aggregator = FedXgbBagging()
            else:
                raise ValueError("Solo se permiten agregadores fedxgb para xgboost: 'fedxgbcyclic', 'fedxgbbagging'")
        elif framework == "pytorch":
            if args.agregator == "fedavg":
                from p2pfl.learning.aggregators.fedavg import FedAvg
                aggregator = FedAvg()
            elif args.agregator == "fedadam":
                from p2pfl.learning.aggregators.fedadam import FedAdam
                aggregator = FedAdam()
            elif args.agregator == "fedadagrad":
                from p2pfl.learning.aggregators.fedadagrad import FedAdagrad
                aggregator = FedAdagrad()
            elif args.agregator == "fedmedian":
                from p2pfl.learning.aggregators.fedmedian import FedMedian
                aggregator = FedMedian()
            elif args.agregator == "fedprox":
                from p2pfl.learning.aggregators.fedprox import FedProx
                aggregator = FedProx()
            elif args.agregator == "fedyogi":
                from p2pfl.learning.aggregators.fedyogi import FedYogi
                aggregator = FedYogi()
            elif args.agregator == "scaffold":
                from p2pfl.learning.aggregators.scaffold import Scaffold
                aggregator = Scaffold()
            elif args.agregator == "krum":
                from p2pfl.learning.aggregators.krum import Krum
                aggregator = Krum()
            else:
                raise ValueError("Agregador no soportado para pytorch")
        else:
            raise ValueError("Framework no soportado. Usa 'xgboost' o 'pytorch'.")

        if framework == "xgboost":
            node_model = XGBoostModel(model=xgboost.XGBClassifier(max_depth=6, n_estimators=1000), id=i)
        elif framework == "pytorch":
            node_model = pytorch_model_build_fn(
                feature_keys=feature_keys,
                label_key=args.target_name,
                label_names=label_names,
                hidden_sizes=[64, 32],
                activation="relu",
                lr_rate=1e-3
            )
        else:
            raise ValueError("Framework no soportado. Usa 'xgboost' o 'pytorch'.")

        node = Node(
            node_model,
            partitions[i],
            protocol=comm_protocol,
            addr=address,
            aggregator=aggregator,
            learner=None
        )
        node.start()
        nodes.append(node)

    try:
        adjacency_matrix = TopologyFactory.generate_matrix(topology, n)
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)
        wait_convergence(nodes, n - 1, only_direct=False, wait=60)
        nodes[0].set_start_learning(rounds=r, epochs=3)  # epochs fijo en 3
        wait_to_finish(nodes, timeout=60 * 60)
    finally:
        for n, node in enumerate(nodes):
            node.stop()
            tracker.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Tabular Classification CLI")
    parser.add_argument('--framework', type=str, default='xgboost', help="Framework: 'xgboost' o 'pytorch'")
    parser.add_argument('--data_csv', type=str, help='Ruta al archivo CSV único para generar train/test')
    parser.add_argument('--train_csv', type=str, help='Ruta al archivo CSV de entrenamiento')
    parser.add_argument('--test_csv', type=str, help='Ruta al archivo CSV de test')
    parser.add_argument('--nodes', type=int, default=3, help='Número de nodos')
    parser.add_argument('--rounds', type=int, default=3, help='Número de rondas')
    parser.add_argument('--seed', type=int, default=42, help='Seed aleatorio')
    parser.add_argument('--topology', type=str, default='LINE', help='Tipo de topología (LINE, RING, STAR, etc)')
    parser.add_argument('--protocol', type=str, default='memory', help="Protocolo de comunicación: 'memory' o 'grpc'")
    parser.add_argument('--target_name', type=str, default='outcome', help='Nombre de la columna objetivo (target)')
    parser.add_argument('--agregator', type=str, default='fedxgbcyclic', help="Agregador a usar. Para xgboost: 'fedxgbcyclic', 'fedxgbbagging'. Para pytorch: 'fedavg', 'fedadam', 'fedadagrad', 'fedmedian', 'fedprox', 'fedyogi', 'scaffold', 'krum'")
    parser.add_argument('--partition_strategy', type=str, default='dirichlet', help="Estrategia de particionado: 'randomiid' o 'dirichlet'")
    parser.add_argument('--dataset', type=str, choices=['ton-iot', 'ugr-16', 'unr-idd', 'unsw-nb15', 'zeekdata'], help="Nombre de un dataset predefinido: data-capec, ton_iot, ugr_16, unr_idd, unsw_nb15 o zeekdata. Se cargará el fichero dataset-NOMBRE.csv de la carpeta correspondiente.")
    args = parser.parse_args()
    run_federated(args)

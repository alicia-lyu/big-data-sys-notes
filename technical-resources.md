# Technical Resources & Critical Code in Assignments

## Manuals

- [The Linux Documentation Project (TLDP)](https://tldp.org) for shell script
- [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

## Linux Commands

- `wget [url] -O [output-file-name]` and `unzip xxx.zip`
- `head --lines -8 path/to/file` to get the first 8 lines; `tail -n 8 path/to/file` to get the last 8 lines
- `|` pipelining, `>` overwrite file with `stdout`, `>>` append `stdout` to file; both `>` and `>>` create a file if non-existent.
- `&>` overwrite file with `stdout` and `stderr` (`&` denotes both `stdout` and `stderr`), the same as `ls > output.txt 2 > output.txt`. We can also do `ls > output.txt 2 > error.txt`, of course.
- `2>&1` redirect `fd 2` (`stderr`) to `fd 1` (`stdout`). `&` indicates that `1` is a fd, not a file name.
- `wc -l/-w/-c path/to/file`: Count lines, words, or characters/bytes
- `grep "search_pattern" path/to/file`: Find patterns in files using regular expressions. `-c` count matches.
- `find root_path -name 'regular_exp'`: Find files or directories under the given directory tree, recursively.
- `[slow_ps] &`: Put the process in the background.
- `ps ax`: processes, `a`: include processes started by other users, `x`: include processes not attached to this terminal.
- `kill pid`: Kill a process by ID
- `pkill "process_name"`: Kill processes which match the name pattern
- `htop`: Display dynamic real-time information about all resources and running processes. An enhanced version of `top`.
- `df`: Display disk usage statistics for file systems
  - `-h`: human readable
- `du path/to/dir`: Display the size of the directory (and its subdirectories)
  - `-h`: human readable, `-s`: a single directory
- `lsof -i tcp -P`: List most port numbers being used by processes on your machine. `lsof` lists open files and the corresponding processes. The `-i` option limits the output to network files, and `tcp` specifies that only TCP protocol connections should be included. `-P` inhibits the conversion of port numbers to port names.
- `ifconfig`: `'lo'` is the loopback interface, only reachable from the local machine, using '`127.0.0.1`.`'eth0'` is the first Ethernet interface, `'wlan0'` is the first wireless interface.

## Docker

docker

- `pull ubuntu:22.04`
- `images`: list images
- `build -t [TAG] .`
- `tag [IMAGE_ID] [TAG]`
- `run [TAG] [COMMAND]`
- `ps -a`: list containers
- `rm [CONTAINER_ID]` v.s. `docker kill [CONTAINER_ID]`: killed container still shows up in `docker ps -a`, while removed container is cleaned up.
- `rmi [IMAGE_ID]`: remove image
- `stats`: get container resource usage statistics
- `system df`: disk usage
- `system prune`: remove unused data
- `logs [CONTAINER_ID]`: get logs
- `exec [CONTAINER_ID] [COMMAND]`: run a command in a running container
- `exec -it [CONTAINER_ID] bash`: enter the container shell. `i` for interactive, `t` for terminal

Dockerfile

- `FROM [IMAGE]`: Base image
- `RUN`: Command to build the application. E.g. `apt update`. Multiple lines:

  ```dockerfile
  RUN <<EOF
  apt-get update
  apt-get install -y wget unzip
  EOF
  ```
- `COPY [PATH_ON_HOST] [PATH_IN_CONTAINER]`: Copy files from the host to the container
- `CMD`: Command to run the application. E.g. `python3 app.py`

Docker compose: `docker compose [OPTIONS] COMMAND`

- `build`
- ` up`
- ` down`: Stop and remove containers, networks
- `kill`: Force remove service containers 

## PyTorch

A self-contained demo for training a simple linear regression model

```python
def train_model(x_tensor, y_tensor): # x_tensor must be initialized with requires_grad=True
    # Define a simple linear regression model
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.DataSet(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(1000):
      for X, Y in iter(dataloader):
          # Forward pass
          outputs = model(X)
          loss = criterion(outputs, Y)

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

    # Print learned parameters
    print(f'Learned parameters: {list(model.parameters())}')
   
def get_tensors(dataname):
  df = pd.read_csv("/nb/%s.csv" % dataname)
  np_array = df.to_numpy(dtype=np.float64)
  arrayX = np_array[:, :-1]
  arrayY = np_array[:, -1:]
  tensorX = torch.from_numpy(arrayX)
  tensorY = torch.from_numpy(arrayY)
  return tensorX, tensorY
```

Get pseudorandom numbers:

```python
torch.manual_seed(0) # Set the seed for generating random numbers
torch.rand(2, 3) # Generate a 2x3 matrix of random numbers
```

Move a CPU tensor A to GPU tensor B: `B = A.to('cuda')`

`torch.sigmoid(x)`: Sigmoid function, `1 / (1 + exp(-x))`, return numbers between 0 and 1.

`A @ B`: Matrix multiplication

```python
model = torch.nn.Linear(x,y)
predictions = model(input)
```

- x: input size, number of features in the input vector, i.e., the number of columns in the input matrix
- y: output size

`torch.cuda.device_count()`

## Python

Create a callable type: initiate a class with `__call__` method

`python3 -u server.py`: force the stdout and stderr streams to be unbuffered

Threading:

```python
t = threading.Thread()
t.start(target=f, args=[x])
# ...
t.get_native_id()
# ...
t.join()
```

Locks:

```python
lock = threading.Lock()

lock.acquire()
# code
lock.release()
```

or

```python
with lock:
    # code with possible exception
```

Ensures lock is released if an exception is thrown inside the block.

File buffering

```python
stations = []
line_len = 86
start = time.time()
with open("ghcnd-stations.txt",
          "rb", buffering=0) as f:
    offset = 0
    while True:
        f.seek(offset)
        station = str(f.read(11), "utf-8") offset += line_len
        if station:
            stations.append(station)
        else: 
            break
print(time.time() - start)
```

`mmap`

- Anonymous
  ```python
  import mmap
  mm = mmap.mmap(-1, 4096*3)
  ```

- Backed by a file
  ```python
  import mmap
  f = open("somefile.txt", mode="rb")
  mm = mmap.mmap(f.fileno(), 0, # 0 means all
          access=mmap.ACCESS_READ)
  ```

## gRPC

Self-contained proto file:

```proto
syntax = "proto3";

package mathdb;

service MathDb {
    rpc Set(SetRequest) returns (SetResponse) {}
}

message SetRequest {
    string key = 1;
    float value = 2;
}

message SetResponse {
    string error = 1;
}
```

`client.py`:

```python
if __name__ == "__main__":
    port, work_files = parse_arguments()
    server_address = "localhost:" + str(port)
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")
    clients = []
    with grpc.insecure_channel(server_address) as channel:
        stub = mathdb_pb2_grpc.MathDbStub(channel)
        threads = []
        for work_file in work_files:
            # Since each client has to wait on IO once in a while
            # I can have more threads than the number of cores
            client = Client(stub, work_file)
            clients.append(client)
            thread = threading.Thread(target=client.main)
            threads.append(thread)
            thread.start()
        print(f'Number of threads: {len(threads)}')
        for thread in threads:
            thread.join()

class Client:
  ...
  
  def main(self):
    ...
    work_iter = self._process_work_file()
    for work in work_iter:
      operation, arg1, arg2 = work
      if operation == "set":
          request = mathdb_pb2.SetRequest(key=arg1, value=arg2)
          response = self.stub.Set(request)
```

`server.py`:

```python
class MathDb(mathdb_pb2_grpc.MathDbServicer):

    def __init__(self):
        self.math_cache = MathCache()

    def Set(self, request, context):
        response = mathdb_pb2.SetResponse()
        try:
            self.math_cache.Set(request.key, request.value)
            response.error = ""
            return response
        except Exception as e:
            response.error = traceback.format_exc()
            return response

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=(('grpc.so_reuseport', 0),))
    mathdb_pb2_grpc.add_MathDbServicer_to_server(MathDb(), server)
    server.add_insecure_port("[::]:5440", )
    server.start()
    server.wait_for_termination()
```

## HDFS Ecosystem

### Parquet

```python
def load_data():
    # Load the Parquet file
    table = pq.read_table('data.parquet')
    # Convert to Pandas DataFrame
    df = table.to_pandas()
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(df[['x']].values, dtype=torch.float32)
    y_tensor = torch.tensor(df[['y']].values, dtype=torch.float32)
    return x_tensor, y_tensor
```

### PyArrow

`read_at(bytes, offset)`

`NativeFile` implements `RawIOBase`.

```python
import pyarrow as pa
import pyarrow.fs as fs
hdfs = fs.HadoopFileSystem("boss", 9000)
with hdfs.open_input_file("/single.csv") as single:
    first_10 = single.read_at(10, 0)
    buffered_reader = io.BufferedReader(single)
    text_wrapper = io.TextIOWrapper(buffered_reader)
    count = sum(1 for line in text_wrapper if "Single Family" in line)
```

### SQL

```sql
-- CREATE TABLE: Create two tables, Authors and Books
CREATE TABLE Authors (
    AuthorID INT PRIMARY KEY,
    Name VARCHAR(100),
    Country VARCHAR(50)
);

CREATE TABLE Books (
    BookID INT PRIMARY KEY,
    Title VARCHAR(100),
    AuthorID INT,
    YearPublished YEAR,
    Genre VARCHAR(50),
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID)
);

-- INSERT: Add entries to both tables
INSERT INTO Authors (AuthorID, Name, Country) VALUES
(1, 'George Orwell', 'UK'),
(2, 'Haruki Murakami', 'Japan'),
(3, 'Jane Austen', 'UK');

INSERT INTO Books (BookID, Title, AuthorID, YearPublished, Genre) VALUES
(1, '1984', 1, '1949', 'Dystopian'),
(2, 'Norwegian Wood', 2, '1987', 'Romance'),
(3, 'Pride and Prejudice', 3, '1813', 'Classic');

-- SELECT: Retrieve all books
SELECT * FROM Books;

-- UPDATE: Update a book's year published
UPDATE Books SET YearPublished = '1948' WHERE BookID = 1;

-- DELETE: Delete an entry
DELETE FROM Books WHERE BookID = 3;

-- ALTER TABLE: Add a new column to Authors
ALTER TABLE Authors ADD BirthYear YEAR;

-- DROP TABLE: (Commented out to preserve data for further commands)
-- DROP TABLE IF EXISTS TableName;

-- CREATE INDEX: Create an index on YearPublished in Books
CREATE INDEX idx_year ON Books(YearPublished);

-- DROP INDEX: (Commented out to maintain the index for demonstration)
-- DROP INDEX idx_year ON Books;

-- SELECT DISTINCT: Select all distinct genres
SELECT DISTINCT Genre FROM Books;

-- WHERE: Select books published after 1940
SELECT * FROM Books WHERE YearPublished > '1940';

-- ORDER BY: Order authors by name
SELECT * FROM Authors ORDER BY Name;

-- GROUP BY and HAVING: Count books per author and filter those with more than 1 book
SELECT AuthorID, COUNT(*) AS NumberOfBooks FROM Books
GROUP BY AuthorID
HAVING COUNT(*) > 1;

-- JOIN (INNER JOIN): Join Books and Authors where they match on AuthorID
SELECT Books.Title, Authors.Name FROM Books
INNER JOIN Authors ON Books.AuthorID = Authors.AuthorID;

-- LEFT JOIN: Select all authors and their books, including authors with no books
SELECT Authors.Name, Books.Title FROM Authors
LEFT JOIN Books ON Authors.AuthorID = Books.AuthorID;

-- RIGHT JOIN: Select all books and their authors, including books with unidentified authors
-- (Assuming there's a possibility for unmatched AuthorID in Books, which is prevented by the foreign key in this setup. Shown for demonstration.)
SELECT Authors.Name, Books.Title FROM Authors
RIGHT JOIN Books ON Authors.AuthorID = Books.AuthorID;

-- UNION: Combine names of all authors and titles of all books (assuming no name-title clashes)
SELECT Name AS NameOrTitle FROM Authors
UNION
SELECT Title FROM Books;

-- LIKE: Find books with titles containing 'wood'
SELECT * FROM Books WHERE Title LIKE '%wood%';

-- IN: Select authors who are from either 'UK' or 'Japan'
SELECT * FROM Authors WHERE Country IN ('UK', 'Japan');

-- Remember to properly manage your database transactions and context to maintain data integrity and performance.
```


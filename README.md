# Pagerank Project

In this project, I created a simple search engine for the website <https://www.lawfareblog.com>, which provides legal analysis on US national security issues.

This project is an exploration of the math concepts in the *Deeper Inside Pagerank* paper (added in this repo). The paper picks up after Sergey Brin and Larry Page's original 1998 paper detailing the algorithm that remains "the heart of [Google’s] software ... and continues to provide the basis for all of [their] web search tools”(http://www.google.com/technology/index.html).

The relevant math for my code is in sections 3 and 5. If this is your first time with Markov Chains, I recommend the first three videos in this short and simple youtube series: https://www.youtube.com/playlist?list=PLM8wYQRetTxBkdvBtz-gw8b9lcVkdXQKV

To summarize the work below: we're creating a web graph of sites as nodes and hyperlinks to create edges. Then, we create the corresponding adjacency matrix (ie P in the paper) and find its eigenvector (the stationary vector of a Markov chain). This stationary vector represents the distribution of the probability of visiting a site after an infinite random walk along the Markov chain. The sites with the highest probability of being visited are most likely to be useful to the user, and therefore, will be returned first! 

## Background

**Some intuition about the data:**  

The data file `lawfareblog.csv.gz` contains the lawfare blog's web graph! 
Let's take a look at the first 10 of these lines:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```
In this file, the node names are URLs.
Semantically, each line corresponds to an HTML `<a>` tag that is contained in the source webpage and links to the target webpage. If we were to draw the graph for the first 4 of these entries, we'd get this: 

<img src="first_4_graph_representation.jpeg" alt="the graph for the first 4" width="300"/>

We can use the following command to count the total number of links/edges in the file:

```
$ zcat data/lawfareblog.csv.gz | wc -l
1610789
```

Every link corresponds to a non-zero entry in the `P` matrix - this is also the value of `nnz(P)`.
(we subtract 1 from this value since the `wc -l` command also counts the header line, not just the data lines.)

To get the dimensions of `P`, we need to count the total number of nodes in the graph, ie the number of unique links.
The following command achieves this by: decompressing the file, extracting the first column, removing all duplicate lines, then counting the results.
```
$ zcat data/lawfareblog.csv.gz | cut -f1 -d, | uniq | wc -l
25761
```
This matrix is large enough that computing matrix products could take several minutes on a single CPU.
Fortunately, however, the matrix is very sparse! This is because a single website will only contain a few hyperlinks - for every website it does NOT link, the entry in that column will be 0!
The following python code computes the fraction of entries in the matrix with non-zero values:
```
>>> 1610788 / (25760**2)
0.0024274297384360172
```
Thus, by using PyTorch's sparse matrix operations, we will be able to speed up the code significantly.

**Code Overview:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases.
For example, you can perform a search for all nodes (i.e. urls) that mention the string `corona` with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --search_query=corona
```

> **NOTE:**
> It will take about 10 seconds to load and parse the data files.
> All the other computation happens essentially instantly.

If you were to comment out the "WebGraph.power_method" function, then the outputted webpages would be returned in an arbitrary order. The following code uses the *Deeper Inside Pagerank* equation 5.1 to calculate which sites are the most important (i.e. have the highest pagerank results) and are returned first.


## Task 0: Code Set-Up, Line-By-Line explanation!
I'm going to go line-by-line through the code now. This is mostly for me because I forget how my code works after like 2 seconds. 

First, we have to make it a little easier to make nodes out of websites. The URLS are superrrr long, lets just number them and reference them by their number! 

Let's create some variables, ` self.url_dict`, `indices`, `target_counts`. 

```
class WebGraph():

    def __init__(self, filename, max_nnz=None, filter_ratio=None):
        '''
        Initializes the WebGraph from a file.
        The file should be a gzipped csv file.
        Each line contains two entries: the source and target corresponding to a single web link.
        This code assumes that the file is sorted on the source column.
        '''

        #actual code 
        self.url_dict = {}
        indices = []
        from collections import defaultdict
        target_counts = defaultdict(lambda: 0)
```
We want the `self.url_dict` as an instance variable - a dictionary specific to each WebGraph object QQ, used to store the mapping of URLs to its new name as a number. Let's say our CSV file is this: 

| source                | target                |
|-----------------------|-----------------------|
| www.example.com        | www.test.com        |
| www.example.com        | www.another.com       |
| www.another.com        | www.example.com       |

Then we want our `self.url_dict` to be a way to refer to our websites with a number instead of its full URL, like so: 

```
#an example 
self.url_dict = {
    'www.example.com': 0,
    'www.test.com': 1,
    'www.another.com': 2
}
```

With an easier way to reference each site, we can treat these numbers as the names of the nodes, and can now start to collect a list of the edges. If website 0 has links to both 1 and 2, and 2 has links to 0, then our `indices` will be:
```
#an example
indices = [[0, 1], [0, 2], [2, 0]]
#instead of ['www.example.com', 'www.test.com', 'www.another.com']
```
We can also start to count how many times a website has been the target, i.e another site has hyperlinked it within itself. 
```
#an example
target_counts = {
    0: 1,
    1: 1,
    2: 1
}
```
Now that we know what those are supposed to look like, let's start using our law blog's info to actually build the corresponding `self.url_dict`, `indices`, and `target_counts`. 

We open up the file and will now go through each row individually in our `for` loop, and use `i` as our counter of what row we are on in case we want to pass in a `max_nnz` that limits how many rows go up to. 

```
#actual code
        # loop through filename to extract the indices
        logging.debug('computing indices')

        #reading the csv file 
        with gzip.open(filename,newline='',mode='rt') as f:
            # i will be the number of the row in the file
            # row will be the row itself 
            for i,row in enumerate(csv.DictReader(f)):

                #creating a limit of rows that it does this for if user specifies a value for max_nnz   
                if max_nnz is not None and i>max_nnz:
                    break
```

This code filters out links with lots of slashes, which usually is directories. 
```
                #this code skips over urls with a lot of slashes - these tend to be directories of links instead of sites themselves!
                import re
                regex = re.compile(r'.*((/$)|(/.*/)).*') #
                if regex.match(row['source']) or regex.match(row['target']):
                    continue
```

Ok, now we are finally building the `target_counts` and `indices`. This is using the WebGraph method `._url_to_index` from below, which is self-explanatory. Recall that our for loop is working on one row at a time. The line `source = self._url_to_index(row['source'])` is just using CSV's indexing format to create/set the variable `source` and `target`  to the left and right columns of the row our `for` loop is working on. With that set, we can update our `target_counts` and `indices`.
```
                source = self._url_to_index(row['source'])
                target = self._url_to_index(row['target'])
                target_counts[target] += 1
                indices.append([source,target])
```

Let's talk about this block of code later; assume that filter_ratio is None for now. 
```
     # remove urls with too many in-links
        if filter_ratio is not None:
            new_indices = []
            for source,target in indices:
                if target_counts[target] < filter_ratio*len(self.url_dict):
                    new_indices.append([source,target])
            indices = new_indices
```

Yay! We've stored the nodes in `self._url_dict`, have the edges of our graph in `indices`, and the in-link number in `target_counts`. Now, we can start computing the transition matrix. Recall that we want to store this as a sparse matrix - specifically, the `torch.sparse.FloatTensor`. 

Sparse matrices assume that most of the numbers in the matrix is 0, and so it only stores the location of the non-zeros and their corresponding values. We know from looking at our .csv that our 0th website will have two edges to the 1st and second. Thus, our matrix will have a 0.5 in the 0th row+1st column and in the 0th row+2nd column. That is specificaly the coordinate information stores as the first two values in `indices = [[0, 1], [0, 2], [2, 0]]` ! 

However, sparse matrices take in the indices with all of the row-coordinates in one list, and the y-coordinates in the other. Thankfully, all that takes is a quick transpose! Storing that transpose in `i`, we get 

```
#an example
i = [[0, 0, 2],   # Sources
     [1, 2, 0]]   # Targets
```

We will store this `i` in the `torch.LongTensor` for compability reasons, so our final code for creating indices:
```
        i = torch.LongTensor(indices).t()
```

We have the indices for our sparse matrix, but we still need to store what values to put in at those spots. This code for me is a little bit of a doozy to explain with words and I think the best way to get an idea of how `values` is beign made is to copy this specific block below into `https://pythontutor.com/visualize.html#mode=edit`: 

```
indices = [[0, 1], [0, 2], [2, 0]]

values = []
last_source = indices[0][0]
last_i = 0
for i,(source,target) in enumerate(indices+[(None,None)]):
    if source==last_source:
        pass
    else:
        total_links = i-last_i
        values.extend([1/total_links]*total_links)
        last_source = source
        last_i = i
```

SLAYYY now we've got both `indices` and `values`. lets put it all together: 

```
        # generate the sparse matrix
        i = torch.LongTensor(indices).t()
        v = torch.FloatTensor(values)
        n = len(self.url_dict)
        self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
```

And finally, we're going to create this dictionary: 
```
        self.index_dict = {v: k for k, v in self.url_dict.items()}
```

YAYYYY we're all set up with our `P` matrix. Now let's start doing some math. 

## The power method

We're using the power method to find the eigenvector ie the PageRank vector of the transition matrix `P`. Go to the paper for an explanation of some of the modifications we make to the raw transition matrix to make it primitive and irreducible and all. The main equation: 

$$ \textbf{x}^{(k)T} = (\alpha \textbf{x}^{(k-1)T})P +  [(\alpha \textbf{x}^{(k-1)T})\textbf{a} + (1 - \alpha)]\textbf{v}^T$$ 

The output of this function $\textbf{x}^{(k)T}$ is the PageRank vector itself! 

We're going to create a function with these inputs: 

```
 def power_method(self, v=None, x0=None, alpha=0.85, max_iterations=1000, epsilon=1e-6):
```

Let's just grab the dimension of the square matrix `P` real quick and store it in `n`: 
```            
        with torch.no_grad():
            n = self.P.shape[0]
```

 
Let's pass in the `v` personalization later - for now, all of the $n$ entries of this vector are going to be $\frac{1}{n}$. We're going to use matrix multiplication instead of matrix/vector multiplication, so we're going to use `torch.unsqueeze`  to add the dimension needed for matrix multiplication. 

```
            # create variables if none given
            if v is None:
                v = torch.Tensor([1/n]*n)
                # v = tensor([0.3333, 0.3333, 0.3333])
                v = torch.unsqueeze(v,1)
                # v = tensor([[0.3333],
                #            [0.3333],
                #            [0.3333]])

            # this line is redudant if we are creating v as above, as it will already be normalized.
            # this line specifically is to make sure that the user-passed personalization vector will work with the rest of the code 
            v /= torch.norm(v)
```

The user may also pass in an `x0` if there is a specific starting point, but otherwise, it will be assumed that there is an equal probabibility of starting on any site. 
```
            if x0 is None:
                x0 = torch.Tensor([1/(math.sqrt(n))]*n)
                x0 = torch.unsqueeze(x0,1)
            x0 /= torch.norm(x0)
```

Now, we need to make the $\textbf{a}$ vector to make $P$ stochastic and all. We know that every row that we've entered numbers into is already stochastic - thus, we just need to check for rows of all 0s. 

```
            stochastic_rows = torch.sparse.sum(self.P,1).indices()
            a = torch.ones([n,1])
            a[stochastic_rows]= 0
```

We have all the variables we need in our code. Now we put it together! We want to map our equation 

$$ \textbf{x}^{(k)T} = (\alpha \textbf{x}^{(k-1)T})P +  [(\alpha \textbf{x}^{(k-1)T})\textbf{a} + (1 - \alpha)]\textbf{v}^T$$ 

to `torch.addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None)` which does: 

out = (beta * input) + (alpha * mat1 * mat2)

OMGG here we go! 
```
            # main loop
            xprev = x0
            x = xprev.detach().clone()
            for i in range(max_iterations):
                xprev = x.detach().clone()

                # compute the new x vector using Eq (5.1)
                # FIXME: Task 1
                # HINT: this can be done with a single call to the `torch.sparse.addmm` function,
                # but you'll have to read the code above to figure out what variables should get passed to that function
                # and what pre/post processing needs to be done to them

                # output debug information
                residual = torch.norm(x-xprev)
                logging.debug(f'i={i} residual={residual}')

                # early stop when sufficient accuracy reached
                if residual < epsilon:
                    break

            #x = x0.squeeze()
            return x.squeeze()


```

Let's test it!! 

The command to run is 
```
python3 pagerank.py --data=data/lawfareblog.csv.gz
```

And this is the output! 

```
% python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
```

Hmmm... a lot of these pages are kind of boring. Lawfare's history, their job board, their subscribe page - all non-article pages. 
We see that our algorithm highly ranks pages with many in-links - without a specific user query, some of the highest-ranked sites are boring non-article pages such the root page <https://lawfareblog.com/>, or a table of contents <https://www.lawfareblog.com/topics>, or a subscribe page <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank, but usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages? The answer is to modify the `P` matrix by removing all links to non-article pages.One easy-to-implement method is to filter nodes by using their "in-link ratio" - the total number of edges with the node as a target (ie this site itself is hyperlinked in other sites) divided by the total number of nodes. Non-article pages often appear in the menu of a webpage, and therefore have links from almost all of the other webpages - thus, their in-link ratio is very high. 

The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than a value that we choose! Let's see the code that does this - we glossed over it the first time we read it. 

```
     # remove urls with too many in-links
        if filter_ratio is not None:
            new_indices = []
            for source,target in indices:
                if target_counts[target] < filter_ratio*len(self.url_dict):
                    new_indices.append([source,target])
            indices = new_indices
```

Let's use the filter ratio parameter, with a chosen ratio cap of 0.2:
```
% python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
These sites look much more like articles than in the previous list!

When Google calculates their `P` matrix for the web,
they use a similar (but much more complicated) process to modify the `P` matrix in order to reduce spam results.
The exact formula they use is a jealously guarded secret that they update continuously.

In the case above, notice that we have accidentally removed the blog's most popular article (<www.lawfareblog.com/snowden-revelations>).
The blog editors believed that Snowden's revelations about NSA spying are so important that they directly put a link to the article on the menu.
So every single webpage in the domain links to the Snowden article,
and our "anti-spam" `--filter-ratio` argument removed this article from the list.
In general, it is a challenging open problem to remove spam from pagerank results,
and all current solutions rely on careful human tuning and still have lots of false positives and false negatives.

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.

**Part 2:**
Now let's see how this works when a user has a specific query. 

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Using the lawfare blog, I get these results:

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9224e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0390e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9153e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7041e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6256e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5046e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3620e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1248e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0187e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=4.5746e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4174e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6928e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9391e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5452e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5357e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4221e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```



**Part 4:**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the `\bar\bar P` matrix,
and that this eigengap is bounded by the alpha parameter.

Run the following four commands:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
You should notice that the last command takes considerably more iterations to compute the pagerank vector.
(My code takes 685 iterations for this call, and about 10 iterations for all the others.)

This raises the question: Why does the second command (with the `--alpha` option but without the `--filter_ratio`) option not take a long time to run?
The answer is that the `P` graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

Changing the value of alpha also gives us very different pagerank rankings.
For example, 
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1755e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best.
If large alphas are good for your application, you can see that there is a trade-off between quality answers and algorithmic runtime.
I'll be exploring this trade-off more formally in my next CS143 projects!

## The personalization vector! 

The most interesting applications of pagerank involve the personalization vector.
Implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _index_to_url function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Part 1:**

The command line argument `--personalization_vector_query` will use the function you created above to augment your search with a custom personalization vector.
If you've implemented the function correctly,
you should get results similar to:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results are significantly different than when using the `--search_query` option:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona'
INFO:root:rank=0 pagerank=8.1320e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=7.7908e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=5.2262e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=3.9584e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=3.8114e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=3.3973e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=3.3633e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=3.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=3.2160e-03 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=3.1036e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

Which results are better?
Again, that depends on what you mean by "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend TONs of time fine-tuning their pagerank personalization vectors to remove spam webpages.
Exactly how they do this is another one of their secrets that they don't publicly talk about.

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.

<!--
**Part 3:**

Select another topic related to national security.
You should experiment with a national security topic other than the coronavirus.
For example, find out what articles are important to the `iran` topic but do not contain the word `iran`.
Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
-->

## Submission

1. Create a new repo on github (not a fork of this repo).

1. Run the following commands, and paste their output into the code blocks below.
   
   Task 1, part 1:
   ```
   $ python3 pagerank.py --data=data/small.csv.gz --verbose
   ```

   Task 1, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
   ```

   Task 1, part 3:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
   ```

   Task 1, part 4:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
   ```

   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
   ```

   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
   ```

1. Ensure that all your changes to the `pagerank.py` and `README.md` files are committed to your repo and pushed to github.

1. Get at least 5 stars on your repo.
   (You may trade stars with other students in the class.)

   > **NOTE:**
   > 
   > Recruiters use github profiles to determine who to hire,
   > and pagerank is used to rank user profiles and projects.
   > Links in this graph correspond to who has starred/followed who's repo.
   > By getting more stars on your repo, you'll be increasing your github pagerank, which increases the likelihood that recruiters will hire you.
   > To see an example, [perform a search for `data mining`](https://github.com/search?q=data+mining).
   > Notice that the results are returned "approximately" ranked by the number of stars,
   > but because "some stars count more than others" the results are not exactly ranked by the number of stars.
   > (I asked you not to fork this repo because forks are ranked lower than non-forks.)
   >
   > In some sense, we are doing a "dual problem" to data mining by getting these stars.
   > Recruiters are using data mining to find out who the best people to recruit are,
   > and we are hacking their data mining algorithms by making those algorithms select you instead of someone else.
   >
   > If you're interested in exploring this idea further, here's a python tutorial for extracting GitHub's social graph: <https://www.oreilly.com/library/view/mining-the-social/9781449368180/ch07.html> ; if you're interested in learning more about how recruiters use github profiles, read this Hacker News post: <https://news.ycombinator.com/item?id=19413348>.

1. Submit the url of your repo to sakai.

   Each part is worth 2 points, for 12 points overall.

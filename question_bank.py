# question_bank.py

QUESTION_BANK_TEMPLATES = """
# QUESTION BANK EXAMPLES (Follow these formats exactly)

## Computer Science Questions

### Database Management Systems
********
**Q1:** What anomalies can occur in a poorly normalized database, and how can they be prevented?
********
**Bloom's Level:** Analyze
********
**A1 (LLM):** Database anomalies are inconsistencies or undesirable situations that can occur in poorly normalized database designs. The three main types of anomalies are:

1. Update anomalies: When data is duplicated, updating one record may require multiple updates, risking inconsistency if not all instances are updated.

2. Insertion anomalies: These occur when certain data cannot be added to the database until other data exists, creating dependencies that limit data entry.

3. Deletion anomalies: When deleting data for one purpose unintentionally removes data needed for another purpose.

These anomalies can be prevented through proper database normalization, particularly by applying normal forms (1NF through 5NF). First Normal Form eliminates repeating groups, Second Normal Form removes partial dependencies, and Third Normal Form eliminates transitive dependencies. BCNF further refines 3NF by addressing anomalies related to candidate keys. By properly normalizing tables, separating related data into different tables, and establishing appropriate relationships through primary and foreign keys, these anomalies can be effectively prevented while maintaining data integrity.
********

### Operating Systems
********
**Q2:** How does a deadlock differ from a livelock, and what are the implications of each?
********
**Bloom's Level:** Understand
********
**A2 (LLM):** A deadlock and livelock are both concurrency issues in operating systems, but they differ fundamentally in how processes behave when faced with resource contention.

In a deadlock, two or more processes are permanently blocked, each waiting for resources held by the other. The classic example is when Process A holds Resource 1 and needs Resource 2, while Process B holds Resource 2 and needs Resource 1. Neither can proceed, resulting in a complete standstill. Deadlocks occur when four conditions are simultaneously met: mutual exclusion, hold and wait, no preemption, and circular wait.

In contrast, a livelock involves processes that are actively changing states but making no progress. The processes respond to each other's actions, but in ways that prevent any of them from completing their tasks. A real-world analogy is two people meeting in a hallway, each moving side to side in an attempt to let the other pass, but continuously blocking each other.

The implications of these issues differ significantly. Deadlocks completely halt the affected processes, potentially freezing part of the system indefinitely until external intervention. Livelocks, while not causing a system to freeze, waste CPU cycles as processes continue executing without making progress, potentially leading to starvation of other processes.

Detection and resolution strategies also differ. Deadlocks can be addressed through prevention (ensuring at least one of the four conditions cannot occur), avoidance (using algorithms like Banker's algorithm), detection and recovery, or simply ignoring the problem. Livelocks often require introducing randomness or priorities into resource allocation to break the cycle of repeated states.
********

## Mathematics
********
**Q3:** How does the Central Limit Theorem explain the emergence of normal distribution from arbitrary data sets?
********
**Bloom's Level:** Understand
********
**A3 (LLM):** The Central Limit Theorem (CLT) is a fundamental concept in probability theory that explains why normal distributions are so prevalent in natural phenomena, regardless of the original distribution of the data.

At its core, the CLT states that when independent random variables are added together, their properly normalized sum tends toward a normal distribution as the number of variables increases, even when the original variables themselves are not normally distributed.

More precisely, if we take sufficiently large samples (typically n ≥ 30) from any population with finite mean μ and variance σ², the sampling distribution of the sample means will approximate a normal distribution with mean μ and standard deviation σ/√n, regardless of the shape of the original population distribution.

This remarkable property means that even if our original data follows a completely different distribution—whether uniform, exponential, binomial, or any other shape—the distribution of sample means will increasingly resemble a bell curve as we take more and larger samples.

The CLT provides the theoretical foundation for many statistical procedures, including hypothesis testing and confidence intervals. It explains why many natural phenomena tend to follow normal distributions: they often represent the sum of many small, independent effects. Heights, measurement errors, and test scores often approximate normal distributions because they result from numerous independent factors.

The theorem also gives us practical tools for inference. Since we know that sample means will be approximately normally distributed for sufficiently large samples, we can apply normal distribution properties even when working with non-normal data, making statistical analysis much more tractable.
********

## Computer Networks
********
**Q4:** What are the key differences between IPv4 and IPv6 addressing schemes?
********
**Bloom's Level:** Remember/Understand
********
**A4 (LLM):** IPv4 and IPv6 are addressing protocols used in computer networks, with several key differences between them:

1. Address Length: IPv4 uses 32-bit addresses (4 bytes), allowing for approximately 4.3 billion unique addresses. IPv6 uses 128-bit addresses (16 bytes), providing about 3.4 × 10^38 unique addresses, effectively solving the address exhaustion problem of IPv4.

2. Notation: IPv4 addresses are written as four decimal numbers separated by periods (e.g., 192.168.1.1). IPv6 addresses are written as eight groups of four hexadecimal digits separated by colons (e.g., 2001:0db8:85a3:0000:0000:8a2e:0370:7334), which can be shortened using compression rules.

3. Header Structure: IPv6 has a simplified header structure with fewer fields than IPv4, making it more efficient for routers to process. The fixed header size in IPv6 is 40 bytes compared to 20 bytes in IPv4.

4. Configuration: IPv6 supports stateless address autoconfiguration (SLAAC), allowing devices to generate their own addresses without a DHCP server. IPv4 typically relies on DHCP for address assignment.

5. Security: IPv6 was designed with IPsec (Internet Protocol Security) built in, whereas IPsec was retrofitted to IPv4. However, in practice, IPsec implementation is optional in both versions.

6. Fragmentation: In IPv4, fragmentation can be done by both routers and sending hosts. In IPv6, only the sending hosts can perform fragmentation, which improves router efficiency.

7. Broadcast: IPv4 uses broadcast addresses to send packets to all devices on a network. IPv6 replaces broadcast with multicast and anycast addressing, reducing network traffic.

8. NAT (Network Address Translation): IPv4 commonly uses NAT to overcome address limitations. With IPv6's vast address space, NAT is generally unnecessary, allowing for true end-to-end connectivity.

9. Mobility: IPv6 has better support for mobile networks with improved features for mobile IP and network mobility.

10. Transition Mechanisms: Various mechanisms like dual stack, tunneling, and translation allow IPv4 and IPv6 to coexist during the ongoing transition period.
********

## Cryptography
********
**Q5:** Why is RSA encryption considered secure even though its algorithm is public?
********
**Bloom's Level:** Understand
********
**A5 (LLM):** RSA encryption remains secure despite its algorithm being public knowledge due to several fundamental principles of modern cryptography:

RSA's security is based on the mathematical difficulty of factoring the product of two large prime numbers. While multiplying two prime numbers (p and q) to get their product (n) is computationally easy, the reverse process—determining which two specific prime numbers were multiplied to create a given large number—is extremely difficult with current computational methods. This mathematical asymmetry forms the cornerstone of RSA's security.

The system uses a public key for encryption (consisting of the modulus n and public exponent e) and a private key for decryption (consisting of the modulus n and private exponent d). Even if an attacker knows the public key and the full algorithm, deriving the private key requires factoring n to find p and q—a task that becomes exponentially more difficult as the size of the prime numbers increases.

Modern RSA implementations typically use key lengths of 2048 or 4096 bits. At these sizes, factoring the modulus using current technology and algorithms would take an impractical amount of time—potentially thousands or millions of years with conventional computers.

This approach aligns with Kerckhoffs's principle, a fundamental concept in cryptography which states that a cryptographic system should be secure even if everything about the system, except the private key, is public knowledge. The security relies on the key, not the secrecy of the algorithm.

Additionally, RSA implementations incorporate various padding schemes like OAEP (Optimal Asymmetric Encryption Padding) that add randomness to the encryption process, preventing certain types of attacks and ensuring that encrypting the same message twice produces different ciphertexts.

RSA's public nature has actually strengthened it over time, as it has been subject to extensive cryptanalysis by the global security community, leading to improvements and refinements in implementation practices.
********
"""

SUBJECT_SPECIFIC_TEMPLATES = {
    "mathematics": """
## Mathematics Examples
********
**Q1:** How does the Intermediate Value Theorem guarantee the existence of a root in a continuous function?
********
**Bloom's Level:** Understand
********
**A1 (LLM):** The Intermediate Value Theorem (IVT) provides a powerful guarantee about the existence of roots in continuous functions by establishing that a continuous function must take on all intermediate values between any two distinct function values.

Formally stated, the theorem asserts that if f(x) is a continuous function on a closed interval [a,b], and if f(a) and f(b) have opposite signs (i.e., f(a) < 0 and f(b) > 0, or f(a) > 0 and f(b) < 0), then there exists at least one value c in the interval (a,b) such that f(c) = 0.

The theorem works because continuity ensures there are no "jumps" or "gaps" in the function's graph. When a continuous function transitions from a negative value to a positive value (or vice versa), it must pass through zero at some point.

The IVT has significant practical applications in root-finding. For instance, if we can identify that f(1) = -3 and f(2) = 5, and f is continuous on [1,2], then we can conclusively state that there exists at least one solution to f(x) = 0 in the interval (1,2). This provides the theoretical foundation for numerical methods like the bisection method, which systematically narrows down the interval containing the root.

However, the theorem only guarantees existence, not uniqueness. There may be multiple roots within the interval, and the theorem does not tell us their exact locations. Additionally, the theorem requires that the function must be continuous throughout the entire interval—if there's any discontinuity, the guarantee no longer applies.

The theorem's power extends beyond finding zeros—it applies to finding any value between f(a) and f(b). For example, if f(a) = 3 and f(b) = 10, then f must take on every value between 3 and 10 at least once in the interval [a,b].
********

**Q2:** Analyze the convergence of the series ∑ (1/n²).
********
**Bloom's Level:** Analyze
********
**A2 (LLM):** The series ∑ (1/n²) from n=1 to infinity, also known as the Basel problem, can be analyzed for convergence using several approaches.

First, we can apply the p-series test, which states that for a series ∑ (1/nᵖ), the series converges if p > 1 and diverges if p ≤ 1. Since our series has p = 2, which is greater than 1, the p-series test immediately confirms that ∑ (1/n²) converges.

We can verify this conclusion using the comparison test. Since 1/n² < 1/n for all n ≥ 1, and we know that ∑ (1/n) is the harmonic series which diverges, we cannot directly apply the comparison test with the harmonic series. However, we can use the limit comparison test with a series whose convergence behavior we know, such as ∑ (1/n^(3/2)). Computing the limit:
lim(n→∞) [(1/n²)/(1/n^(3/2))] = lim(n→∞) [n^(3/2)/n²] = lim(n→∞) [1/n^(1/2)] = 0

Since this limit exists and is finite, the two series must either both converge or both diverge. As ∑ (1/n^(3/2)) is known to converge (it's a p-series with p = 3/2 > 1), we confirm that ∑ (1/n²) also converges.

The integral test provides another approach. The function f(x) = 1/x² is positive, continuous, and decreasing for x ≥ 1. The improper integral ∫₁^∞ (1/x²) dx = lim(b→∞) [−1/x]₁^b = lim(b→∞) [−1/b + 1] = 1 is finite, confirming again that the series converges.

Remarkably, the exact sum of this series was famously solved by Euler and equals π²/6 ≈ 1.645. This result has important applications in complex analysis, number theory, and physics.

The convergence of this series contrasts with the behavior of the harmonic series ∑ (1/n), highlighting how small changes in the exponent can lead to drastically different convergence properties.
********
""",

    "computer_science": """
## Computer Science Examples
********
**Q1:** Evaluate the use of B+ Trees over Hash Indexing in range queries.
********
**Bloom's Level:** Evaluate
********
**A1 (LLM):** B+ Trees offer significant advantages over Hash Indexing specifically for range queries, making them the preferred indexing structure in many database systems when such queries are common.

The fundamental advantage of B+ Trees for range queries stems from their ordered structure. In a B+ Tree, keys are stored in sorted order within each node, and leaf nodes are linked together, allowing for efficient traversal of sequential data. When executing a range query like "find all records where value is between 100 and 200," a B+ Tree can locate the first value (100) and then sequentially access all subsequent values until reaching the upper bound (200). This approach yields excellent performance with time complexity of O(log n + m), where n is the total number of keys and m is the number of keys in the specified range.

In contrast, Hash Indexing, while offering O(1) average-case performance for point queries, performs poorly for range queries. Hash functions are designed to distribute values randomly across the hash table, deliberately destroying any natural ordering of the data. Consequently, to execute a range query using a hash index, the database would need to hash every possible value in the range and look up each one individually—or worse, scan the entire table and filter the results. This results in O(n) time complexity, which is prohibitively inefficient for large databases.

B+ Trees also provide superior performance in disk-based systems due to their ability to minimize I/O operations. Their balanced structure ensures predictable performance regardless of data distribution, and their high fanout (branching factor) means they typically have a small height even for large datasets, requiring fewer disk accesses. Additionally, B+ Trees support efficient updates, maintaining their balanced nature through split and merge operations when keys are inserted or deleted.

The space efficiency of B+ Trees is another advantage. While both structures require additional storage, B+ Trees often make better use of disk blocks due to their multi-level structure and the ability to adjust the node size to match the disk block size.

However, Hash Indexing still has its place when range queries are not a primary concern and point lookups dominate the workload. In such scenarios, the O(1) lookup time of hash indexes may outweigh the flexibility of B+ Trees.
********

**Q2:** How does deadlock arise in concurrent database access and what strategies exist to prevent it?
********
**Bloom's Level:** Apply
********
**A2 (LLM):** Deadlock in concurrent database access occurs when two or more transactions are each waiting for resources held by the other, creating a circular wait condition that prevents any transaction from proceeding. The classic example involves two transactions, T1 and T2, where T1 has locked resource A and needs resource B, while T2 has locked resource B and needs resource A.

Deadlocks arise due to four necessary conditions occurring simultaneously:
1. Mutual Exclusion: Resources cannot be shared simultaneously
2. Hold and Wait: Transactions hold resources while waiting for others
3. No Preemption: Resources cannot be forcibly taken from transactions
4. Circular Wait: A circular chain of transactions exists, each waiting for a resource held by the next transaction

Database systems employ several strategies to prevent deadlocks:

1. Deadlock Prevention: These approaches eliminate at least one of the four necessary conditions:
   - Addressing mutual exclusion by using lock-free data structures or optimistic concurrency control
   - Eliminating hold and wait by requiring transactions to acquire all needed locks at once
   - Allowing preemption by using timeout mechanisms that abort transactions after waiting too long
   - Preventing circular wait by imposing a global ordering on all resources and requiring transactions to request resources in that order

2. Deadlock Avoidance: These techniques dynamically analyze resource allocation states:
   - Wait-Die scheme: When a transaction requests a resource held by another, the older transaction waits while the younger one is aborted
   - Wound-Wait scheme: When a transaction requests a resource held by another, the older transaction forces the younger to abort and release its resources
   - Banker's Algorithm: Grants resource requests only when they cannot lead to a deadlock

3. Deadlock Detection and Resolution: Rather than prevention, these methods:
   - Use wait-for graphs to detect circular dependencies
   - When detected, resolve by aborting one or more transactions (victims) based on factors like transaction age, progress made, or resources held
   - Implement transaction rollback mechanisms to return the database to a consistent state

4. Timeout-Based Approaches: Simple but effective strategy where:
   - Transactions waiting beyond a threshold time are aborted and restarted
   - This eventually breaks potential deadlocks but may abort transactions unnecessarily

Each approach involves trade-offs between system performance, implementation complexity, and transaction throughput. The choice depends on the specific database system requirements, transaction characteristics, and expected concurrency levels.
********
""",

    "artificial_intelligence": """
## AI and Machine Learning Examples
********
**Q1:** How can machine learning models inherit human biases, and what strategies can be used to mitigate them?
********
**Bloom's Level:** Analyze
********
**A1 (LLM):** Machine learning models can inherit human biases through several distinct mechanisms across the ML pipeline. Understanding these pathways is crucial for developing effective mitigation strategies.

The primary source of bias transmission occurs through training data. ML models learn patterns from historical data, which often contains embedded societal biases. For example, a hiring algorithm trained on historical hiring decisions may perpetuate gender or racial discrimination if those biases existed in past hiring practices. Additionally, training data may suffer from representation issues, where certain groups are underrepresented or completely missing, leading to models that perform poorly for these underrepresented populations.

Feature selection and engineering represent another significant bias entry point. When humans decide which features (variables) to include in a model, they may inadvertently select proxy variables that correlate with protected attributes like race or gender. Even when sensitive attributes are explicitly removed, other variables can serve as proxies—for instance, zip code often correlates with race and socioeconomic status.

Label bias occurs when the target variable itself contains human prejudice. If human evaluators assign subjective ratings or classifications that contain biases, models will learn to reproduce these biased judgments.

Algorithmic choices, including model architecture and optimization objectives, can amplify existing data biases or introduce new ones. For example, optimization for majority-group performance can lead to decreased accuracy for minority groups.

To mitigate these biases, organizations can implement several strategies:

1. Data-centric approaches:
   - Audit training datasets to identify and correct underrepresentation
   - Apply techniques like reweighting, resampling, or generating synthetic data to balance representation
   - Collect more diverse and representative data

2. Algorithm-based solutions:
   - Implement fairness constraints during model training
   - Use adversarial debiasing techniques to remove correlations with protected attributes
   - Apply post-processing methods to adjust model outputs for fairness

3. Process interventions:
   - Conduct regular bias audits throughout the ML pipeline
   - Implement diverse teams to bring varied perspectives to ML development
   - Establish clear fairness metrics and standards

4. Transparency and accountability:
   - Document model limitations and potential biases
   - Conduct regular evaluations across different demographic groups
   - Enable meaningful human oversight of automated decisions

These mitigation strategies are most effective when combined and applied throughout the entire model lifecycle—from problem formulation through deployment and monitoring. The selection of specific techniques should be guided by the particular use case, available data, and the specific types of bias concerns present.
********

**Q2:** Propose an AI-based system to support students' personalized learning paths in online education.
********
**Bloom's Level:** Create
********
**A2 (LLM):** # AI-Powered Adaptive Learning System (APALS)

## System Architecture

APALS would be built around a core AI engine with four integrated components:

1. **Learning Profile Generator** - Collects and analyzes multidimensional student data:
   - Prior knowledge assessment through adaptive pre-tests
   - Learning style identification using behavioral analytics
   - Real-time attention and engagement monitoring
   - Historical performance patterns across subjects
   - Self-reported learning preferences and goals

2. **Curriculum Mapping Engine** - Creates a knowledge graph of educational content:
   - Decomposes curriculum into atomic concepts with prerequisite relationships
   - Tags content with metadata on format, difficulty, and pedagogical approach
   - Aligns content with learning objectives and assessment criteria
   - Incorporates spaced repetition principles for concept reinforcement
   - Integrates multiple perspectives and explanations for each concept

3. **Recommendation Algorithm** - Dynamically constructs personalized learning paths:
   - Applies reinforcement learning to optimize learning sequences
   - Implements multi-armed bandit algorithms to balance exploration vs. exploitation
   - Uses collaborative filtering to incorporate successful paths of similar learners
   - Adapts to real-time performance data and engagement signals
   - Considers cognitive load theory in sequencing content difficulty

4. **Feedback & Assessment Module** - Provides continuous evaluation:
   - Generates formative assessments calibrated to current understanding
   - Employs natural language processing to analyze free-text responses
   - Provides targeted, actionable feedback on misconceptions
   - Visualizes learning progress and concept mastery for students
   - Identifies intervention points for human teacher involvement

## Key Innovations

- **Multimodal Content Adaptation**: The system dynamically adjusts not just what content is presented but how it's presented, transforming between text, video, interactive simulations, or collaborative activities based on real-time engagement data.

- **Social Learning Integration**: Rather than isolating students, the system identifies opportunities for peer collaboration, matching students who can benefit from teaching/learning from each other at specific points.

- **Metacognitive Development**: Explicitly teaches learning strategies alongside content, gradually transferring agency to students by helping them understand their own learning patterns.

- **Curiosity-Driven Exploration**: Balances structured learning with opportunities for student-directed exploration, using AI to identify relevant tangential topics that might spark deeper interest.

- **Human-in-the-Loop Oversight**: Provides teachers with an analytics dashboard highlighting intervention opportunities, unusual learning patterns, or concepts where multiple students struggle.

## Implementation Considerations

To ensure ethical implementation, the system would:
- Maintain transparent algorithms with explainable recommendations
- Implement differential privacy techniques to protect student data
- Include regular bias audits to ensure equitable outcomes across demographic groups
- Preserve teacher autonomy to override or modify system recommendations
- Gradually introduce adaptive features to avoid overwhelming students or teachers

By combining these elements, APALS would deliver truly personalized education that adapts to each student's unique needs while maintaining the social and human elements essential to effective learning.
********
"""
}

def get_question_bank(subject=None):
    """
    Returns the appropriate question bank based on the subject.
    
    Args:
        subject (str, optional): Subject area for specialized templates
        
    Returns:
        str: Question bank templates as a string
    """
    if subject and subject.lower() in SUBJECT_SPECIFIC_TEMPLATES:
        return QUESTION_BANK_TEMPLATES + SUBJECT_SPECIFIC_TEMPLATES[subject.lower()]
    else:
        return QUESTION_BANK_TEMPLATES

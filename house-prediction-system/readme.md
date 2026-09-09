about data
Let's decode the dataset

Dataset has 2,930 houses and 82 columns. The columns fall broadly into these groups:

Group	Examples	What they describe
Identification	Order, PID	Identifies the house
Location	Neighborhood, MS Zoning	Where/type of area
Land	Lot Area, Lot Frontage, Lot Shape	Property land
Quality	Overall Qual, Overall Cond	Overall house quality/condition
Age	Year Built, Year Remod/Add	House age/history
Basement	Total Bsmt SF, Bsmt Qual	Basement
Living space	Gr Liv Area, 1st Flr SF, 2nd Flr SF	House size
Rooms	Full Bath, Bedroom AbvGr, TotRms AbvGrd	Rooms/bathrooms
Garage	Garage Cars, Garage Area, Garage Qual	Garage
Outdoor	Wood Deck SF, Open Porch SF	Outdoor space
Amenities	Fireplaces, Pool Area	Features
Sale information	Mo Sold, Yr Sold, Sale Type	Transaction
Target	SalePrice	What we predict

But let's go deeper, because this is where you'll actually learn ML feature engineering.

1. Identification columns
Order

Example:

1
2
3
4
...
2930

This is essentially the row/order number.

It doesn't describe the house.

Likely decision: REMOVE

PID

Example:

526301100
526350040
526351010

PID = Parcel Identification Number.

It identifies a specific property.

Again, it isn't a physical characteristic of the house.

Likely decision: REMOVE from the model.

However, we shouldn't blindly delete it before checking whether it encodes geographic information. We'll investigate that.

2. Land/property characteristics
MS SubClass

This is a building class/type code.

It's numerical:

20
30
60
...

But here's an important ML lesson:

A number doesn't necessarily mean a numerical quantity.

For example:

MS SubClass = 20
MS SubClass = 60

doesn't mean that 60 is "three times" 20.

It's a category represented by a number.

So we'll probably treat this as categorical.

MS Zoning

Zoning classification.

Examples:

RL
RH
RM
...

This describes the zoning of the property.

Categorical.

Lot Frontage

Linear feet of the property connected to the street.

Example:

141
80
81
93

Numerical.

Missing values exist.

Potentially useful.

Lot Area

Total land area.

Example:

31770
11622
14267
11160

Numerical.

Likely useful.

Street

Type of road access.

For example:

Pave
Grvl

Categorical.

Alley

Type of alley access.

Many houses don't have alley access, which explains why you have a huge number of missing values.

This is an important point:

NaN

doesn't necessarily mean:

"Data was accidentally forgotten."

It can mean:

"This house doesn't have an alley."

That's something we'll handle carefully during preprocessing.

3. Land shape / geography
Lot Shape

Shape of the lot.

Categorical.

Land Contour

Flatness/topography of the property.

Categorical.

Utilities

Available utilities.

Categorical.

Lot Config

Configuration of the property.

Categorical.

Land Slope

Slope of the land.

Categorical.

These are potentially useful because property characteristics influence price.

4. Location
Neighborhood

This is a VERY important feature.

It tells us which neighborhood the house is in.

For example:

NAmes
CollgCr
OldTown
Edwards
...

This can have a strong relationship with price.

You should not remove this just because it's categorical.

In fact, after One-Hot Encoding:

Neighborhood = CollgCr

becomes something like:

Neighborhood_CollgCr = 1

This allows the model to learn different price patterns between neighborhoods.

5. House type/design

These columns describe the physical/design characteristics.

Condition 1

Nearby conditions affecting the property.

Condition 2

Secondary nearby condition.

Bldg Type

Building type.

House Style

Style of the house.

Roof Style

Roof type.

Roof Matl

Roof material.

Exterior 1st

Primary exterior material.

Exterior 2nd

Secondary exterior material.

These are categorical features.

Some may be useful, some may turn out to have very little predictive value.

We'll measure rather than guess.

6. The most important group: quality
Overall Qual

Overall material and finish quality.

This is probably one of the most important features in your dataset.

You've already discovered:

Correlation with SalePrice = 0.799

That's extremely strong.

Conceptually:

Higher quality
       ↓
Higher expected price
Overall Cond

Overall condition of the house.

Notice the difference:

Overall Qual
     ↓
Quality of materials/finish

Overall Cond
     ↓
Current/general condition

They sound similar but represent different concepts.

Your correlation showed:

Overall Cond ≈ -0.102

That's interesting.

Don't immediately remove it.

We'll investigate why.

7. Age
Year Built

Year the house was originally constructed.

Example:

1960
2005
2010

Very useful.

But here's where feature engineering becomes interesting.

Instead of only:

Year Built

we can create:

HouseAge = Yr Sold - Year Built

Example:

Year Built = 2000
Yr Sold = 2010

HouseAge = 10

This is much more directly related to the property's age at the time of sale.

Year Remod/Add

Year of remodeling/addition.

We can potentially create:

RemodelAge = Yr Sold - Year Remod/Add

Again, this represents something more meaningful than the raw year.

8. Masonry
Mas Vnr Type

Masonry veneer type.

Categorical.

Mas Vnr Area

Masonry veneer area.

Numerical.

These describe brick/stone veneer.

9. Basement

Now look at these:

Bsmt Qual
Bsmt Cond
Bsmt Exposure
BsmtFin Type 1
BsmtFin SF 1
BsmtFin Type 2
BsmtFin SF 2
Bsmt Unf SF
Total Bsmt SF
Bsmt Full Bath
Bsmt Half Bath

They describe different aspects of the basement.

For example:

Total Bsmt SF

Total basement square footage.

Very useful.

You found:

Correlation ≈ 0.632
BsmtFin SF 1

Finished basement area, type 1.

BsmtFin SF 2

Finished basement area, type 2.

Bsmt Unf SF

Unfinished basement area.

This gives us an interesting possibility:

Total Bsmt SF
      ↓
already contains basement area information

So we need to investigate whether some of these variables are redundant.

That's feature selection.

10. Living area

These are extremely important:

1st Flr SF
2nd Flr SF
Gr Liv Area
Low Qual Fin SF
Gr Liv Area

Above-ground living area.

You found:

Correlation ≈ 0.707

Very strong.

We can potentially create:

TotalAboveGroundSF =
    1st Flr SF + 2nd Flr SF

But we have to think carefully because Gr Liv Area may already overlap heavily with those variables.

So rather than blindly create:

TotalSF

we'll investigate their relationships first.

11. Bathrooms

You have:

Full Bath
Half Bath
Bsmt Full Bath
Bsmt Half Bath

Instead of treating all four independently, we could create:

TotalBathrooms

For example:

Full Bath = 2
Half Bath = 1
Bsmt Full Bath = 1
Bsmt Half Bath = 0

TotalBathrooms
= 2 + 0.5 + 1
= 3.5

That's a very reasonable engineered feature.

12. Bedrooms and rooms
Bedroom AbvGr
Kitchen AbvGr
TotRms AbvGrd

These describe room counts.

But again, we should investigate redundancy.

For example:

Gr Liv Area
       ↕
TotRms AbvGrd
       ↕
Bedroom AbvGr

These aren't independent.

13. Kitchen
Kitchen AbvGr
Kitchen Qual

Kitchen Qual is categorical quality.

This could be very predictive.

A house with:

Excellent Kitchen

may command a higher price than one with:

Poor Kitchen
14. Garage

This group is important:

Garage Type
Garage Yr Blt
Garage Finish
Garage Cars
Garage Area
Garage Qual
Garage Cond

You already found:

Garage Cars ≈ 0.648
Garage Area ≈ 0.640

Notice something interesting:

Garage Cars
      ↕
Garage Area

They're measuring related things.

This is exactly the type of relationship we'll investigate.

Maybe both should remain.

Maybe one is redundant.

Maybe an engineered feature is better.

We will test it.

15. Outdoor space
Wood Deck SF
Open Porch SF
Enclosed Porch
3Ssn Porch
Screen Porch

These are square-foot measurements of outdoor areas.

We can potentially create:

TotalPorchSF

because individually they may have relatively weak relationships, but collectively they represent usable outdoor space.

16. Special amenities
Fireplaces
Fireplace Qu
Pool Area
Pool QC
Fence
Misc Feature
Misc Val

These describe special features.

Notice your missing values:

Pool QC       2917 missing
Misc Feature  2824 missing
Fence         2358 missing

This tells us something.

For example:

Pool QC = NaN

may actually mean:

House has no pool

rather than:

Unknown pool quality

That's a semantic missing value.

This is one of the most important things you'll learn in real-world ML.

17. Sale information
Mo Sold
Yr Sold
Sale Type
Sale Condition

These describe the transaction.

We need to be careful here.

For example:

Yr Sold

may contain useful market information.

But it also raises a question:

Are we building a model to predict the price of a house at the time it is being sold, or are we trying to estimate its value today?

That definition affects which features we should use.

For this project, we'll define the prediction problem explicitly before finalizing the feature set.

18. SalePrice

This is our target:

SalePrice

Everything else should be evaluated based on:

Can this information legitimately help us predict SalePrice?